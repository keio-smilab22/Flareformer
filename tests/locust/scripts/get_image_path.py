"""画像パス取得APIの負荷試験"""
import http
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
    def get_image_path(self):
        """日付から画像パスを取得する"""
        rand_dates = random.choice(date_list)
        with self.client.get(f"/images/path?date={rand_dates}", catch_response=True) as response:
            if response.status_code != http.HTTPStatus.OK.value:
                response.failure(f"StatusCode is not 200 but {response.status_code}")
            else:
                try:
                    content = json.loads(response.content.decode())
                    if content["get_image_status"] == "failed":
                        response.failure("Failed to get image path list")
                except json.JSONDecodeError as error:
                    response.failure(f"Failed to decode the response(discribed below) as json:\n{error.doc}")
