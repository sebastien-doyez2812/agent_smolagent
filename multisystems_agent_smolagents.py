import math 
import os
from PIL import Image
from typing import Optional, Tuple
from smolagents import tool
from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, LiteLLMModel
from smolagents.utils import encode_image_base64, make_image_url

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2",
    api_key = "ollama"
)


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """
    def to_radians(degrees: float) -> float:
        return degrees * (math.pi/180)
    
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    EARTH_RADIUS_KM = 6371.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    actual_distance = distance * 1.1

    # Calculate flight time
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    return round(flight_time, 2)


def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = LiteLLMModel(model_id= "ollama_chat/llava")
    filepath = "saved_map.png"
    assert os.path.exists(filepath)
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]

    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True


# agent = CodeAgent(
#     model=model,
#     tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_cargo_travel_time],
#     additional_authorized_imports=["pandas"],
#     max_steps=20,
# )

web_agent = CodeAgent(
    model = model,
    tools= [DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_cargo_travel_time],
    name = "web_agent",
    description = "Browse the web to find information",
    verbosity_level = 0,
    max_steps = 10
)


manager_agent = CodeAgent(
    model=LiteLLMModel( model_id= "ollama_chat/deepseek-r1", max_tokens=8096),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)


manager_agent.visualize()
# task = """
# Step 1: Find a list of Batman filming locations and their coordinates.
# Step 2: Calculate the travel time from (40.7128, -74.0060) to each filming location.
# Step 3: Return the results as a Pandas DataFrame.
# """

# # task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
# # Also give me some supercar factories with the same cargo plane transfer time."""

# print(agent.run(task))