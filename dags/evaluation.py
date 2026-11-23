"""Independent DAG: evaluation can succeed even when forecasting fails."""
import datetime as dt
import pendulum

from airflow.sdk import dag, task
from app.pipeline_tasks import update_actuals_and_scores

NY = pendulum.timezone("America/New_York")


@dag(
    dag_id="evaluation",
    schedule="0 20 * * *",  # example: 8:00 pm local, adjust to preference
    start_date=pendulum.datetime(2026, 1, 1, tz=NY),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "nyc-electricity", "retries": 2,
                  "retry_delay": dt.timedelta(minutes=15)},
    tags=["electricity", "evaluation"],
)
def evaluation_dag():
    @task
    def refresh_and_score(**context) -> dict[str, object]:
        day = context["data_interval_end"].in_timezone(NY).date()
        return update_actuals_and_scores(day)

    refresh_and_score()


evaluation_dag()
