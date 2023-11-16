from locust import HttpUser, between, task


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
    def _post_test(self, url):
        with open("dog.jpeg", "rb") as img:
            self.client.post(url, files={"file": img})

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.

    #### Set task(3) for this test and 2 for the others. Why? Because in real apps
    #### many users just visit the landing page without doing any other action
    @task(3)
    def index_get(self):
        self.client.get("/")

    @task(2)
    def index_post(self):
        self._post_test("/")

    @task(2)
    def predict(self):
        self._post_test("/predict")
