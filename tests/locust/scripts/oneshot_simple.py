"""一発打ちAPIの負荷試験"""
import json
import random

from locust import HttpUser, constant, task

with open("tests/locust/config/requestable_queries.json", "r") as f:
    date_json = json.load(f)
    date_list = date_json["dates"]


class WebsiteUser(HttpUser):
    """HTTPシステムの負荷テストを行うクラス"""

    wait_time = constant(60)

    @task
    def oneshot_simple(self):
        """日付から一発打ちの結果を取得する"""
        rand_dates = random.choice(date_list)
        with self.client.get(f"/oneshot/simple?date={rand_dates}", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"StatusCode is not 200 but {response.status_code}")
            else:
                try:
                    content = json.loads(response.content.decode())
                    if content["oneshot_status"] == "failed":
                        response.failure("oneshot status is failed")
                except json.JSONDecodeError as error:
                    response.failure(f"JSONDecodeError: {error.doc}")
