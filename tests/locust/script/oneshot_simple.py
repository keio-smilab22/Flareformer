"""一発打ちAPIの負荷試験"""
import json
import random

from locust import HttpUser, constant, task

with open("tests/locust/config/sample_dates.json", "r") as f:
    date_json = json.load(f)
    date_list = date_json["date_list"]


class WebsiteUser(HttpUser):
    """HTTPシステムの負荷テストを行うクラス"""

    wait_time = constant(60)

    @task
    def oneshot_simple(self):
        """日付から一発打ちの結果を取得する"""
        rand_dates = random.choice(date_list)
        with self.client.get(f"/oneshot/simple?date={rand_dates}", catch_response=True) as response:
            content = json.loads(response.content.decode())
            if content["oneshot_status"] == "failed":
                response.failure("oneshot status is failed")
            elif response.status_code != 200:
                response.failure("statusCode is not 200")
