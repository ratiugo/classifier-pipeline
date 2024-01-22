create_environment: .FORCE
	conda env remove --name gp
	conda env create -f environment.yml

save_env: .FORCE
	conda env export > environment.yml

test: .FORCE
	python -m unittest discover -v -b

lint: .FORCE
	black src test classifiers --preview --exclude classifiers/spotify_genre_predictor/fma_utils.py 
	pylint src test classifiers --disable=import-error

auth: .FORCE
	PYTHONPATH=. python spotify_auth.py

spotify_data: .FORCE
	PYTHONPATH=. python classifiers/spotify_genre_predictor/prepare_fma_dataset.py\
		--config-file-name classifiers/spotify_genre_predictor/input/config.json

train_classifier: .FORCE
	./train_classifier.sh \
		--config-file-name "classifiers/spotify_genre_predictor/input/config.json"

spotify_predict_dataset: .FORCE
	PYTHONPATH=. python classifiers/spotify_genre_predictor/generate_predict_dataset.py \
		--spotify-user-name ratiugo \
		--playlist-name Jid \
		--config-file-name classifiers/spotify_genre_predictor/input/config.json

.FORCE: