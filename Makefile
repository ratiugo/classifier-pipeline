create_environment: .FORCE
	conda env remove --name gp
	conda env create -f environment.yml

save_env: .FORCE
	conda env export > environment.yml

test: .FORCE
	python -m unittest discover -v -b

lint: .FORCE
	black src test --preview --exclude src/fma_utils.py 
	pylint src test --disable=import-error

auth: .FORCE
	PYTHONPATH=. python spotify_auth.py

spotify_data: .FORCE
	PYTHONPATH=. python classifiers/spotify_genre_predictor/prepare_fma_dataset.py

classifier: .FORCE
	./train_classifier.sh \
		--data-file-name "classifiers/spotify_genre_predictor/input/data.csv" \
		--classifier-config-file-name "classifiers/spotify_genre_predictor/input/config.json"

.FORCE: