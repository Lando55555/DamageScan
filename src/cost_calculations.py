from llm_interface import query_llm_for_cost_calculation

def calculate_costs(e3_dry_times):
    """
    Calculate costs using the LLM based on E3 drying times.
    
    Parameters:
    e3_dry_times (dict): Drying times per room.
    
    Returns:
    dict: Costs broken down by labor, materials, and equipment.
    
    Raises:
    Exception: If the LLM API call fails.
    """
    try:
        costs = query_llm_for_cost_calculation(e3_dry_times, "http://0.0.0.0:1234/v1/chat/completions", "mathstral-7b-v0.1-q4_k_m")
        return costs
    except Exception as e:
        raise Exception(f"Error in LLM cost calculation: {str(e)}")
