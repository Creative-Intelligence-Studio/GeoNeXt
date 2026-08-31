from diffsynth.core import UnifiedDataset
import os 
class GeoNeXtDataset(UnifiedDataset):
    def __init__(self, data_profile,base_path, metadata_path, *args, **kwargs):
        self.profile = data_profile
        self._bad_sample_count = 0
        # Call the parent class, but disable metadata loading there.
        super().__init__(metadata_path=None, *args, **kwargs)
        self.load_from_cache = False
        # Core behavior: let the profile load and transform dataset entries.
        full_meta_path = metadata_path
             
        # 1) Load normalized data entries via profile.
        self.data = self.profile.load_and_transform(full_meta_path)
        
        # 2) Register profile-defined operators.
        self.special_operator_map = self.profile.get_operator_map()
        self.data_file_keys = self.profile.get_data_keys()
        
        # 3) Disable cache-read mode for online preprocessing.
        
        print(f"[Dataset] Initialized with profile: {self.profile.__class__.__name__}")
        print(f"[Dataset] Total samples: {len(self.data)}")

    # Override parent metadata loader (unused in profile-driven mode).
    def load_metadata(self, metadata_path):
        pass

    def __getitem__(self, data_id):
        try:
            if self.load_from_cache:
                data = self.cached_data[data_id % len(self.cached_data)]
                data = self.cached_data_operator(data)
            else:
                data = self.data[data_id % len(self.data)].copy()
                for key in self.data_file_keys:
                    if key in data:
                        if key in self.special_operator_map:
                            data[key] = self.special_operator_map[key](data[key])
                        elif key in self.data_file_keys:
                            data[key] = self.main_data_operator(data[key])
            if hasattr(self.profile, "postprocess_sample"):
                data = self.profile.postprocess_sample(data)
            return data
        except Exception as exc:
            # Robust fallback: skip corrupted samples instead of crashing DDP training.
            self._bad_sample_count += 1
            if self._bad_sample_count <= 20 or self._bad_sample_count % 100 == 0:
                print(
                    f"[Dataset][Warn] Skip bad sample data_id={data_id} "
                    f"(count={self._bad_sample_count}): {exc}"
                )
            return None


# Backward-compatible alias.
FlexibleDataset = GeoNeXtDataset
