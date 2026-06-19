import os

import h5py
import numpy as np


class Hdf5CoordinateOutput:
    extension = ".h5"

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def output_path(self, slide_id):
        return os.path.join(self.output_dir, slide_id + self.extension)

    def exists(self, slide_id, expected_count=None):
        return os.path.isfile(self.output_path(slide_id))

    def write(self, slide_id, asset_dict, attr_dict=None, mode="a"):
        output_path = self.output_path(slide_id)
        with h5py.File(output_path, mode) as file:
            for key, val in asset_dict.items():
                data_shape = val.shape
                if key not in file:
                    data_type = val.dtype
                    chunk_shape = (1,) + data_shape[1:]
                    maxshape = (None,) + data_shape[1:]
                    dset = file.create_dataset(
                        key,
                        shape=data_shape,
                        maxshape=maxshape,
                        chunks=chunk_shape,
                        dtype=data_type,
                    )
                    dset[:] = val
                    if attr_dict is not None and key in attr_dict:
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
                else:
                    dset = file[key]
                    dset.resize(len(dset) + data_shape[0], axis=0)
                    dset[-data_shape[0]:] = val
        return output_path


class LanceCoordinateOutput:
    extension = ""

    def __init__(self, db_path, table_name="wsi_patch_coords"):
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError(
                "LanceDB output requires lancedb and pyarrow. Install them to use "
                "--output_adapter lance."
            ) from exc

        self.db_path = db_path
        self.table_name = table_name
        self._pa = pa
        self.db = lancedb.connect(db_path)
        self.table = self._open_or_create_table()
        self._patch_idx = {}

    def _open_or_create_table(self):
        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)

        schema = self._pa.schema(
            [
                ("slide_id", self._pa.string()),
                ("patch_idx", self._pa.int64()),
                ("coord_x", self._pa.int64()),
                ("coord_y", self._pa.int64()),
                ("patch_size", self._pa.int64()),
                ("patch_level", self._pa.int64()),
                ("downsample_x", self._pa.float64()),
                ("downsample_y", self._pa.float64()),
                ("level_width", self._pa.int64()),
                ("level_height", self._pa.int64()),
            ]
        )
        return self.db.create_table(self.table_name, schema=schema)

    def output_path(self, slide_id):
        return f"{self.db_path}::{self.table_name}/{slide_id}"

    def exists(self, slide_id, expected_count=None):
        try:
            count = self.table.search().where(f"slide_id = '{slide_id}'").to_arrow().num_rows
        except Exception:
            values = self.table.to_lance().to_table(columns=["slide_id"]).column("slide_id").to_pylist()
            count = sum(value == slide_id for value in values)
        if expected_count is not None:
            return count >= expected_count
        return count > 0

    def write(self, slide_id, asset_dict, attr_dict=None, mode="a"):
        coords = asset_dict.get("coords")
        if coords is None or len(coords) == 0:
            return self.output_path(slide_id)
        if mode == "w":
            try:
                self.table.delete(f"slide_id = '{slide_id}'")
            except Exception:
                pass
            self._patch_idx[slide_id] = 0

        attrs = (attr_dict or {}).get("coords", {})
        patch_size = int(attrs.get("patch_size", 0))
        patch_level = int(attrs.get("patch_level", 0))
        downsample = attrs.get("downsample", (1.0, 1.0))
        level_dim = attrs.get("level_dim", (0, 0))
        start_idx = self._patch_idx.get(slide_id, 0)

        records = []
        for offset, coord in enumerate(np.asarray(coords)):
            records.append(
                {
                    "slide_id": slide_id,
                    "patch_idx": start_idx + offset,
                    "coord_x": int(coord[0]),
                    "coord_y": int(coord[1]),
                    "patch_size": patch_size,
                    "patch_level": patch_level,
                    "downsample_x": float(downsample[0]),
                    "downsample_y": float(downsample[1]),
                    "level_width": int(level_dim[0]),
                    "level_height": int(level_dim[1]),
                }
            )

        self.table.add(records)
        self._patch_idx[slide_id] = start_idx + len(records)
        return self.output_path(slide_id)


def build_patch_output(output_adapter, patch_save_dir, lance_db_path=None, lance_table_name="wsi_patch_coords"):
    if output_adapter == "hdf5":
        return Hdf5CoordinateOutput(patch_save_dir)
    if output_adapter == "lance":
        db_path = lance_db_path or patch_save_dir
        return LanceCoordinateOutput(db_path, table_name=lance_table_name)
    raise ValueError(f"Unsupported output adapter: {output_adapter}")


class Hdf5CoordinateInput:
    def __init__(self, data_h5_dir):
        self.data_h5_dir = data_h5_dir

    def load(self, slide_id):
        file_path = os.path.join(self.data_h5_dir, "patches", slide_id + ".h5")
        with h5py.File(file_path, "r") as file:
            dset = file["coords"]
            coords = dset[:]
            attrs = dict(dset.attrs.items())
        return coords, attrs


class LanceCoordinateInput:
    def __init__(self, db_path, table_name="wsi_patch_coords"):
        try:
            import lancedb
        except ImportError as exc:
            raise ImportError(
                "LanceDB coordinate input requires lancedb. Install it to use "
                "--coords_adapter lance."
            ) from exc

        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)

    def load(self, slide_id):
        try:
            df = self.table.search().where(f"slide_id = '{slide_id}'").to_pandas()
        except Exception:
            df = self.table.to_pandas()
            df = df[df["slide_id"] == slide_id]

        if df.empty:
            raise FileNotFoundError(f"No LanceDB coordinates found for slide_id={slide_id}")

        if "patch_idx" in df:
            df = df.sort_values("patch_idx")
        coords = df[["coord_x", "coord_y"]].to_numpy(dtype=np.int32)

        first = df.iloc[0]
        attrs = {
            "patch_size": int(first.get("patch_size", 0)),
            "patch_level": int(first.get("patch_level", 0)),
            "downsample": (
                float(first.get("downsample_x", 1.0)),
                float(first.get("downsample_y", 1.0)),
            ),
            "level_dim": (
                int(first.get("level_width", 0)),
                int(first.get("level_height", 0)),
            ),
            "name": slide_id,
        }
        return coords, attrs


class Hdf5FeatureOutput:
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.h5_dir = os.path.join(feat_dir, "h5_files")
        self.pt_dir = os.path.join(feat_dir, "pt_files")
        os.makedirs(self.h5_dir, exist_ok=True)
        os.makedirs(self.pt_dir, exist_ok=True)

    def output_path(self, slide_id):
        return os.path.join(self.h5_dir, slide_id + ".h5")

    def exists(self, slide_id, expected_count=None):
        return os.path.isfile(os.path.join(self.pt_dir, slide_id + ".pt"))

    def write_batch(self, slide_id, features, coords, patch_indices, mode="a"):
        from utils.file_utils import save_hdf5

        asset_dict = {"features": features, "coords": coords}
        save_hdf5(self.output_path(slide_id), asset_dict, attr_dict=None, mode=mode)
        return self.output_path(slide_id)

    def finalize_slide(self, slide_id):
        import torch

        output_path = self.output_path(slide_id)
        with h5py.File(output_path, "r") as file:
            features = file["features"][:]
            coords_shape = file["coords"].shape
            print("features size: ", features.shape)
            print("coordinates size: ", coords_shape)

        torch.save(
            torch.from_numpy(features),
            os.path.join(self.pt_dir, slide_id + ".pt"),
        )
        return output_path


class LanceFeatureOutput:
    def __init__(self, db_path, table_name="wsi_patch_embeddings", model_name=None):
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError(
                "LanceDB feature output requires lancedb and pyarrow. Install them "
                "to use --feature_output_adapter lance."
            ) from exc

        self.db_path = db_path
        self.table_name = table_name
        self.model_name = model_name or ""
        self._pa = pa
        self.db = lancedb.connect(db_path)
        self.table = self._open_or_create_table()

    def _open_or_create_table(self):
        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)

        schema = self._pa.schema(
            [
                ("slide_id", self._pa.string()),
                ("patch_idx", self._pa.int64()),
                ("coord_x", self._pa.int64()),
                ("coord_y", self._pa.int64()),
                ("embedding", self._pa.list_(self._pa.float32())),
                ("model_name", self._pa.string()),
            ]
        )
        return self.db.create_table(self.table_name, schema=schema)

    def output_path(self, slide_id):
        return f"{self.db_path}::{self.table_name}/{slide_id}"

    def exists(self, slide_id, expected_count=None):
        try:
            count = self.table.search().where(f"slide_id = '{slide_id}'").to_arrow().num_rows
        except Exception:
            df = self.table.to_pandas()
            count = int((df["slide_id"] == slide_id).sum())
        if expected_count is not None:
            return count >= expected_count
        return count > 0

    def write_batch(self, slide_id, features, coords, patch_indices, mode="a"):
        if mode == "w":
            try:
                self.table.delete(f"slide_id = '{slide_id}'")
            except Exception:
                pass

        records = []
        for feature, coord, patch_idx in zip(features, coords, patch_indices):
            records.append(
                {
                    "slide_id": slide_id,
                    "patch_idx": int(patch_idx),
                    "coord_x": int(coord[0]),
                    "coord_y": int(coord[1]),
                    "embedding": np.asarray(feature, dtype=np.float32).tolist(),
                    "model_name": self.model_name,
                }
            )
        if records:
            self.table.add(records)
        return self.output_path(slide_id)

    def finalize_slide(self, slide_id):
        print("features written to: ", self.output_path(slide_id))
        return self.output_path(slide_id)


def build_coord_input(coords_adapter, data_h5_dir=None, lance_db_path=None, lance_table_name="wsi_patch_coords"):
    if coords_adapter == "hdf5":
        return Hdf5CoordinateInput(data_h5_dir)
    if coords_adapter == "lance":
        return LanceCoordinateInput(lance_db_path, table_name=lance_table_name)
    raise ValueError(f"Unsupported coordinate input adapter: {coords_adapter}")


def build_feature_output(
    feature_output_adapter,
    feat_dir,
    lance_db_path=None,
    lance_table_name="wsi_patch_embeddings",
    model_name=None,
):
    if feature_output_adapter == "hdf5":
        return Hdf5FeatureOutput(feat_dir)
    if feature_output_adapter == "lance":
        db_path = lance_db_path or feat_dir
        return LanceFeatureOutput(db_path, table_name=lance_table_name, model_name=model_name)
    raise ValueError(f"Unsupported feature output adapter: {feature_output_adapter}")
