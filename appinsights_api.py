import logging

import requests

key = "lfxwlb5d15eda5cuy3pa4gl2p47c8re2klylm3r8"
appId = "c2041544-aa9e-4919-8226-e417951e20eb"
api_url = f"https://api.applicationinsights.io/v1/apps/{appId}/query"


def call_api(query):
    # query = "traces | take 10"
    query_params = {"query": query}
    headers = {"x-api-key": key}
    response = requests.get(api_url, params=query_params, headers=headers)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print("HTTP error occurred: ", err)
        print("The query that caused the error: ", query)
    except Exception as err:
        print("Other error occurred: ", err)
    else:
        return response.json()
