import math
import json
from typing import Dict, Any, List
from pydantic import BaseModel

class E3CalculatorInput(BaseModel):
    weather_data: Dict[str, float]
    room_data: List[Dict[str, Any]]

class E3CalculatorTool:
    name: str = "E3DryTimeCalculator"
    description: str = "Calculates E3 dry times for rooms based on weather data and room specifications."

    def calculate_e3_dry_times(self, data: Dict[str, Any]) -> Dict[str, Any]:
        weather_data = data['weather_data']
        room_data = data['room_data']
        
        # Calculate psychrometric properties
        dew_point = self.calculate_dew_point(weather_data['temp_c'], weather_data['humidity'])
        wet_bulb = self.calculate_wet_bulb(weather_data['temp_c'], weather_data['humidity'])
        gpp = self.calculate_gpp(weather_data['temp_c'], weather_data['humidity'], weather_data['pressure_mb'])
        
        # Calculate energies
        ambient_energy = self.calculate_energy(weather_data['temp_f'], gpp)
        dew_point_energy = self.calculate_energy((dew_point * 9/5) + 32, gpp)
        wet_bulb_energy = self.calculate_energy((wet_bulb * 9/5) + 32, gpp)
        
        # Calculate E³ value
        e3_value = self.calculate_e3(ambient_energy, dew_point_energy, wet_bulb_energy)
        
        results = []
        total_area = 0
        weighted_e3_sum = 0
        
        for room in room_data:
            room_size = eval(room['Size_of_Room'].replace('x', '*'))
            total_area += room_size
            room_e3 = e3_value * self.get_airflow_factor(room['Natural_Airflow'])
            dry_time = self.calculate_room_dry_time(room, room_e3)
            weighted_e3_sum += room_e3 * room_size
            results.append({
                'Room_ID': room['Room_ID'],
                'Room_Name': room['Room_Name'],
                'Room_E3': round(room_e3, 2),
                'Estimated_Dry_Time': dry_time,
                'Room_Size': room_size,
                'Weight_Percentage': 0  # Will be calculated after total area is known
            })
        
        for room in results:
            room['Weight_Percentage'] = round((room['Room_Size'] / total_area) * 100, 2)
        
        weighted_average_e3 = weighted_e3_sum / total_area
        weighted_average_dry_time = self.calculate_weighted_average_dry_time(results)
        
        return {
            'room_results': results,
            'weighted_average_e3': round(weighted_average_e3, 2),
            'weighted_average_dry_time': weighted_average_dry_time,
            'total_area': total_area
        }

    def calculate_dew_point(self, temp_c: float, rh: float) -> float:
        a = 17.27
        b = 237.7
        alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
        return (b * alpha) / (a - alpha)

    def calculate_wet_bulb(self, temp_c: float, rh: float) -> float:
        temp_f = (temp_c * 9/5) + 32
        wet_bulb_f = temp_f * math.atan(0.151977 * math.sqrt(rh + 8.313659)) + \
                     math.atan(temp_f + rh) - math.atan(rh - 1.676331) + \
                     0.00391838 * math.sqrt(rh**1.5) * math.atan(0.023101 * rh) - 4.686035
        return (wet_bulb_f - 32) * 5/9

    def calculate_gpp(self, temp_c: float, rh: float, pressure_mb: float) -> float:
        es = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
        e = (rh / 100.0) * es
        w = 0.622 * (e / (pressure_mb - e))
        return w * 7000

    def calculate_energy(self, temp_f: float, gpp: float) -> Dict[str, float]:
        sensible = temp_f * 0.24 + (gpp / 7000) * 0.45 * temp_f
        latent = (gpp / 7000) * 1061
        total = sensible + latent
        return {'sensible': sensible, 'latent': latent, 'total': total}

    def calculate_e3(self, ambient_energy: Dict[str, float], dew_point_energy: Dict[str, float], wet_bulb_energy: Dict[str, float]) -> float:
        fw = (ambient_energy['sensible'] - wet_bulb_energy['sensible']) * 10
        bw = (ambient_energy['sensible'] - dew_point_energy['sensible']) * 5
        return fw + bw

    def get_airflow_factor(self, natural_airflow: str) -> float:
        num_openings = len(natural_airflow.split(','))
        return 1 + (num_openings - 1) * 0.1  # 10% increase for each additional opening

    def calculate_room_dry_time(self, room: Dict[str, Any], room_e3: float) -> Dict[str, float]:
        base_time = {1: 3, 2: 5, 3: 7}[room['Damage_Class']]
        room_size = eval(room['Size_of_Room'].replace('x', '*'))
        
        dry_time_days = (base_time * room_size / 100) / (room_e3 / 100)
        dry_time_hours = dry_time_days * 24
        return {
            'days': round(dry_time_days, 1),
            'hours': round(dry_time_hours, 1)
        }

    def calculate_weighted_average_dry_time(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total_weighted_time = 0
        total_weight = 0
        for room in results:
            weight = room['Weight_Percentage'] / 100
            total_weighted_time += room['Estimated_Dry_Time']['days'] * weight
            total_weight += weight
        avg_days = total_weighted_time / total_weight
        return {
            'days': round(avg_days, 1),
            'hours': round(avg_days * 24, 1)
        }

    def __call__(self, input_data: E3CalculatorInput) -> str:
        results = self.calculate_e3_dry_times(input_data.dict())
        
        output = "Estimated Dry Times and E3 Values:\n"
        output += "| Room ID | Room Name | E3 Value | Dry Time (Days) | Dry Time (Hours) | Weight % |\n"
        output += "|---------|-----------|----------|-----------------|------------------|----------|\n"
        for room in results['room_results']:
            output += f"| {room['Room_ID']} | {room['Room_Name']} | {room['Room_E3']} | {room['Estimated_Dry_Time']['days']} | {room['Estimated_Dry_Time']['hours']} | {room['Weight_Percentage']}% |\n"
        
        output += f"\nWeighted Average E3 for the entire site: {results['weighted_average_e3']}"
        output += f"\nWeighted Average Dry Time: {results['weighted_average_dry_time']['days']} days ({results['weighted_average_dry_time']['hours']} hours)"
        
        return output

# Function to generate JSON output
def generate_json_output(results):
    return json.dumps(results, indent=2)

# Main function to process user input and run calculations
def process_user_input(user_input: str) -> str:
    try:
        # Parse the user input JSON
        input_data = json.loads(user_input)
        
        # Validate input data using Pydantic model
        validated_input = E3CalculatorInput(**input_data)
        
        # Create E3 Calculator instance
        e3_calculator = E3CalculatorTool()
        
        # Run calculations
        results = e3_calculator.calculate_e3_dry_times(validated_input.dict())
        
        # Generate output
        table_output = e3_calculator(validated_input)
        json_output = generate_json_output(results)
        
        return f"{table_output}\n\nJSON Output:\n{json_output}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Please check your input format."
    except ValueError as e:
        return f"Error: {str(e)}"

# Example usage: This would typically be replaced by actual user input
if __name__ == "__main__":
    user_input = '''
    {
      "weather_data": {
        "temp_c": 25,
        "temp_f": 77.0,
        "humidity": 55,
        "pressure_mb": 1015.0
      },
      "room_data": [
        {
          "Room_ID": 1,
          "Room_Name": "Living Room",
          "Size_of_Room": "20x18",
          "Natural_Airflow": "Window, Door",
          "Damage_Class": 2
        },
        {
          "Room_ID": 2,
          "Room_Name": "Bedroom 1",
          "Size_of_Room": "15x14",
          "Natural_Airflow": "Window",
          "Damage_Class": 1
        }
      ]
    }
    '''
    
    output = process_user_input(user_input)
    print(output)
