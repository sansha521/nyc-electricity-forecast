"""Daily NY-local forecast, with EIA publication lag handled by a sensor."""
import datetime as dt
import pendulum

from airflow.sdk import dag, task
from airflow.providers.standard.sensors.python import PythonSensor

from app.pipeline_tasks import (
    ingest_latest_eia,
    validate_data_availability,
    build_feature_row,
    generate_and_store_forecast,
)

NY = pendulum.timezone("America/New_York")


def demand_ready_sensor(**context) -> bool:
    target = context["ti"].xcom_pull(task_ids="date_for_run")
    return validate_data_availability(dt.date.fromisoformat(target))


@dag(
    dag_id="forecast",
    schedule="00 06 * * *",  # 6:00 am New York time, DST-aware
    start_date=pendulum.datetime(2026, 1, 1, tz=NY),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "nyc-electricity", "retries": 2,
                  "retry_delay": dt.timedelta(minutes=10)},
    tags=["electricity", "forecast"],
)
def forecast_dag():
    @task
    def ingest(target_date: str) -> int:
        return ingest_latest_eia(dt.date.fromisoformat(target_date))

    @task
    def build(target_date: str) -> dict[str, float]:
        return build_feature_row(dt.date.fromisoformat(target_date))

    @task
    def predict_and_save(target_date: str, features: dict[str, float]) -> float:
        return generate_and_store_forecast(dt.date.fromisoformat(target_date), features)

    @task
    def date_for_run(**context) -> str:
        # data_interval_end is the scheduled wall-time boundary; don't use
        # datetime.now() in retries or manual historical runs.
        return context["data_interval_end"].in_timezone(NY).date().isoformat()

    target = date_for_run()
    ingested = ingest(target)
    ready = PythonSensor(
        task_id="validate_data_availability",
        python_callable=demand_ready_sensor,
        poke_interval=15 * 60,
        timeout=4 * 60 * 60,
        mode="reschedule",
    )
    features = build(target)
    prediction = predict_and_save(target, features)
    ingested >> ready >> features >> prediction # Task dependencies


forecast_dag()
