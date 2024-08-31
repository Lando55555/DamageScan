import requests
import json
from custom_exceptions import LLMQueryError
import logging

logging.basicConfig(level=logging.INFO)

def query_llm_for_cost_calculation(e3_dry_times, llm_url, model_name):
    """
    Queries an LLM to calculate costs based on E3 drying times.

    Parameters:
    e3_dry_times (dict): The drying times for each room.
    llm_url (str): The URL of the LLM service.
    model_name (str): The name of the LLM model to use.

    Returns:
    dict: A dictionary containing labor, material, and equipment costs.

    Raises:
    LLMQueryError: If the LLM API request or response parsing fails.
    """
    headers = {"Content-Type": "application/json"}
    prompt = f"Calculate costs based on these drying times: {json.dumps(e3_dry_times)}. Provide labor, material, and equipment costs."

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a cost estimation expert for water damage restoration."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "labor_cost": {"type": "number"},
                    "material_cost": {"type": "number"},
                    "equipment_cost": {"type": "number"}
                },
                "required": ["labor_cost", "material_cost", "equipment_cost"]
            }
        },
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        logging.info("Querying LLM for cost calculation")
        response = requests.post(llm_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return json.loads(result['choices'][0]['message']['content'])
    except requests.RequestException as e:
        logging.error(f"LLM API request failed: {str(e)}")
        raise LLMQueryError(f"LLM API request failed: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        logging.error(f"Error parsing LLM response: {str(e)}")
        raise LLMQueryError(f"Error parsing LLM response: {str(e)}")
