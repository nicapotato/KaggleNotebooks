# Keras / TensorFlow notebooks – M3 Mac

SITE_PKG    := .venv/lib/python3.12/site-packages
# Bazel rpath that libmetal_plugin.dylib was compiled with (removed in newer TF builds)
SOLIB_DIR   := $(SITE_PKG)/_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf

# Sync dependencies then patch the missing Metal rpath symlink
.PHONY: sync
sync:
	uv sync
	mkdir -p $(SOLIB_DIR)
	ln -sf $(CURDIR)/$(SITE_PKG)/tensorflow/python/_pywrap_tensorflow_internal.so \
	       $(SOLIB_DIR)/_pywrap_tensorflow_internal.so

# Launch Jupyter server (single target)
.PHONY: notebook
notebook:
	uv run jupyter notebook .

# uv run script targets
.PHONY: run-customer-lifetime-value run-timeseries-classification run-timeseries-weather
run-customer-lifetime-value:
	MPLBACKEND=Agg uv run python tutorials/customer_lifetime_value.py

run-timeseries-classification:
	MPLBACKEND=Agg KERAS_TUTORIAL_QUICK=1 uv run python tutorials/timeseries_classification_from_scratch.py

run-timeseries-weather:
	MPLBACKEND=Agg uv run python tutorials/timeseries_weather_forecasting.py

# uv run nbconvert targets – clear outputs, execute notebook top-to-bottom, write results in-place
.PHONY: nbconvert-customer-lifetime-value nbconvert-keras-getting-started nbconvert-timeseries-classification nbconvert-snp
nbconvert-customer-lifetime-value:
	uv run jupyter nbconvert --execute --to notebook --inplace \
		--ClearOutputPreprocessor.enabled=True tutorials/customer_lifetime_value.ipynb

nbconvert-keras-getting-started:
	uv run jupyter nbconvert --execute --to notebook --inplace \
		--ClearOutputPreprocessor.enabled=True tutorials/keras_getting_started.ipynb

nbconvert-timeseries-classification:
	uv run jupyter nbconvert --execute --to notebook --inplace \
		--ClearOutputPreprocessor.enabled=True tutorials/timeseries_classification_from_scratch.ipynb

nbconvert-snp:
	uv run jupyter nbconvert --execute --to notebook --inplace \
		--ClearOutputPreprocessor.enabled=True standards-poor-darkmatter/SP500-Multi-Output-DarkMatter-25Y.ipynb

# Smoke test (synthetic data; no kagglehub download)
.PHONY: nbconvert-snp-debug
nbconvert-snp-debug:
	SNP_DEBUG_SYNTHETIC=1 SNP_EPOCHS=3 SNP_BATCH=32 uv run jupyter nbconvert --execute --to notebook --inplace \
		--ClearOutputPreprocessor.enabled=True standards-poor-darkmatter/SP500-Multi-Output-DarkMatter-25Y.ipynb
