from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_SETTINGS = {
    "DATABASE_URL": "postgresql://test:test@localhost/test",
    "EIA_API_KEY": "test-key",
    "VISUAL_CROSSING_API_KEY": "test-key",
}


@unittest.skipUnless(find_spec("airflow"), "Install apache-airflow[standard] for DAG tests")
class AirflowDagTests(unittest.TestCase):
    def check_dags(self, checks, *, worker_settings=False):
        airflow_home = self.enterContext(TemporaryDirectory())
        env = {
            key: value for key, value in os.environ.items()
            if key not in WORKER_SETTINGS and not key.startswith("AIRFLOW_")
        }
        env.update({
            "AIRFLOW_HOME": airflow_home,
            "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
            "PYTHON_DOTENV_DISABLED": "1",
        })
        if worker_settings:
            env.update(WORKER_SETTINGS)

        load_dags = dedent('''
            from airflow.dag_processing.dagbag import DagBag

            bag = DagBag(dag_folder="dags")
            assert not bag.import_errors, bag.import_errors
            assert {"forecast", "evaluation"} <= bag.dags.keys(), bag.dags.keys()
        ''')
        completed = subprocess.run(
            [sys.executable, "-B", "-c", load_dags + dedent(checks)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_scheduler_can_import_dags_without_worker_credentials(self):
        self.check_dags("")

    def test_forecast_sensor_gates_prediction_and_evaluation_has_its_own_schedule(self):
        self.check_dags('''
            from datetime import date
            from unittest.mock import Mock, call, patch

            from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
            from airflow.sdk import CronDataIntervalTimetable, CronTriggerTimetable

            forecast = bag.dags["forecast"]
            sensor = forecast.get_task("validate_data_availability")
            assert sensor.mode == "reschedule", sensor.mode
            assert "ingest" in sensor.upstream_task_ids, sensor.upstream_task_ids
            assert sensor.task_id in forecast.get_task("build").upstream_task_ids
            assert "build" in forecast.get_task("predict_and_save").upstream_task_ids

            ti = Mock()
            ti.xcom_pull.return_value = "2026-11-23"
            validate = Mock(side_effect=[False, True])
            with patch.dict(sensor.python_callable.__globals__, validate_data_availability=validate):
                assert not sensor.poke({"ti": ti})
                assert sensor.poke({"ti": ti})
            assert validate.call_args_list == [call(date(2026, 11, 23))] * 2

            evaluation = bag.dags["evaluation"]
            assert isinstance(evaluation.timetable, (CronTriggerTimetable, CronDataIntervalTimetable))
            assert evaluation.tasks
            assert not any(isinstance(task, ExternalTaskSensor) for task in evaluation.tasks)
        ''', worker_settings=True)

    def test_forecast_run_date_uses_new_york_day_from_run_context(self):
        self.check_dags('''
            from types import SimpleNamespace
            import pendulum

            scheduled = pendulum.datetime(2026, 7, 11, 2, 30, tz="UTC")
            task = bag.dags["forecast"].get_task("date_for_run")
            for try_number in (1, 2):
                target = task.python_callable(
                    data_interval_end=scheduled,
                    dag_run=SimpleNamespace(run_after=scheduled),
                    ti=SimpleNamespace(try_number=try_number),
                )
                assert target == "2026-07-10", target
        ''', worker_settings=True)
