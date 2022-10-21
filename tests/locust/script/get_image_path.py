from locust import HttpUser, task, constant
import json
import random


with open("tests/locust/config/sample_dates.json", "r") as f:
    date_json = json.load(f)
    date_list = date_json["date_list"]

class WebsiteUser(HttpUser):
    wait_time = constant(60)

    @task
    def get_image_path(self):        
        rand_dates = random.choice(date_list)
        with self.client.get(f"/images/path?date={rand_dates}", catch_response=True) as response:
            content = json.loads(response.content.decode())
            if content["get_image_status"] == "failed":
                response.failure("get image status is failed")
            elif response.status_code != 200:
                response.failure("statusCode is not 200")
