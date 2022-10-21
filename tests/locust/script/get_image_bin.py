from locust import HttpUser, task, constant
import json
import random


with open("tests/locust/config/sample_image_paths.json", "r") as f:
    date_json = json.load(f)
    image_path_list = date_json["image_path_list"]

class WebsiteUser(HttpUser):
    wait_time = constant(60)

    @task
    def get_image_bin(self):
        rand_image_path = random.choice(image_path_list)
        with self.client.get(f"/images/bin?path={rand_image_path}", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("statusCode is not 200")
