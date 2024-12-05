# Practice 17. Building a recommendation scenario in the production system
## 💻 Hardware Requirements
The solution works stably even with 4 GB of RAM, except for the last step with serving the model and sending requests to it. Therefore, it is better to use 8 GB of RAM or more.

## 🚀 How to start this masterpiece
0. If you are using Windows OS then install [WSL]( https://learn.microsoft.com/ru-ru/windows/wsl/install)
1. Install [docker](https://docs.docker.com/engine/install/)
2. Login to docker to pull image `echo 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 | docker login ghcr.io -u USERNAME --password-stdin`
3. Infrastructure preparation using `docker compose up airflow-init`
4. Run docker compose by `docker compose up`. Please note the terminal should not be interrupted for the correct operation of the entire pipeline.
5. Forward ports `8085`, `5000`, `5006` to localhost. Port 8085 will be used by AirFlow, port 5000 will be used by MLFlow, port 5006 will be used to access the model via the REST API.
6. Go to http://localhost:8085 in AirFlow and login using following credentials: login - `airflow`, password - `airflow`.
7. Run the `Sasrec_model_DAG` dag. After DAG's work is completed, the SasRec transformer model from the RePlay library on the Movielens-100k dataset will be trained. All necessary data will be saved to the `volumes` folder during the tasks operations. The trained model will be uploaded to `MLFlow` and ready for serving.
8. Go to the terminal. Create env with `Python 3.10`, install `mlflow==2.7.0` and `replay-rec[torch]==0.16.0`.
9. To serve the model, run the following command `MLFLOW_TRACKING_URI='http://localhost:5000' mlflow models serve --env-manager local -m models:/sasrec/latest -p 5006`. Please note the terminal should not be interrupted for the correct operation of the entire pipeline.
10. Now, to receive recommendations from the model, you need to send a POST request to port 5006 with the content of user interactions which are sorted in ascending order of interaction time. As an example, you can use the following request directly from the terminal. `curl -X POST -H "Content-Type: application/json" --data '{"inputs": [1, 2, 3, 4, 5]}' http://localhost:5006/invocations`. This means that the user interacted with the item #5 last. The behavior of the model at this stage is regulated in the file `models/sasrec/model.py`.
