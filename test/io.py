import numpy as np
from lxml import etree
import pandas as pd
def generate_osc(vehicle_data: dict, filename: str = "example_trajectory.xosc"):
    """
    Generates an OpenSCENARIO file with trajectories for multiple vehicles, including entities and a storyboard.

    Parameters:
    - vehicle_data: dict, where each key is a vehicle name (str) and each value is a dictionary containing:
        - 'x': list of the x-coordinates of the trajectory.
        - 'y': list of the y-coordinates of the trajectory.
        - 'z': list of the z-coordinates of the trajectory.
        - 'yaw': list of the heading (yaw) angles along the trajectory.
        - 'pitch': list of the pitch angles along the trajectory.
        - 'roll': list of the roll angles along the trajectory.
        - 'timestamp': list of the timestamps corresponding to each position along the trajectory.

    - filename: str, the name of the output OpenSCENARIO file.

    Output:
    - An OpenSCENARIO file stored at the specified filename.
    """
    # Create the root element for the OpenSCENARIO file
    openscenario = etree.Element("OpenSCENARIO")

    # Add the Entities section
    entities = etree.SubElement(openscenario, "Entities")
    for vehicle_name in vehicle_data.keys():
        scenario_object = etree.SubElement(entities, "ScenarioObject", name=vehicle_name)
        vehicle = etree.SubElement(scenario_object, "Vehicle", name=vehicle_name, vehicleCategory="car")
        etree.SubElement(vehicle, "ParameterDeclarations")

    # Add the Storyboard section
    storyboard = etree.SubElement(openscenario, "Storyboard")

    # Add the Init section (initialization of the scenario)
    init = etree.SubElement(storyboard, "Init")
    actions = etree.SubElement(init, "Actions")

    for vehicle_name in vehicle_data.keys():
        private_action = etree.SubElement(actions, "Private", entityRef=vehicle_name)
        private_position_action = etree.SubElement(private_action, "PrivateAction")
        teleport_action = etree.SubElement(private_position_action, "TeleportAction")
        position = etree.SubElement(teleport_action, "Position")
        etree.SubElement(position, "WorldPosition", 
                         x=str(vehicle_data[vehicle_name]['x'][0]), 
                         y=str(vehicle_data[vehicle_name]['y'][0]), 
                         z=str(vehicle_data[vehicle_name]['z'][0]),
                         h=str(vehicle_data[vehicle_name]['yaw'][0]),
                         p=str(vehicle_data[vehicle_name]['pitch'][0]),
                         r=str(vehicle_data[vehicle_name]['roll'][0]),
                         )

    # Add the Story section with a Maneuver for each vehicle
    story = etree.SubElement(storyboard, "Story", name="MainStory")
    act = etree.SubElement(story, "Act", name="MainAct")
    maneuver_group = etree.SubElement(act, "ManeuverGroup", maximumExecutionCount="1", name="ManeuverGroup1")

    for vehicle_name in vehicle_data.keys():
        maneuver = etree.SubElement(maneuver_group, "Maneuver", name=f"{vehicle_name}_Maneuver")
        event = etree.SubElement(maneuver, "Event", name="Event1", priority="overwrite")
        action = etree.SubElement(event, "Action", name="Action1")
        private_action = etree.SubElement(action, "PrivateAction")
        routing_action = etree.SubElement(private_action, "RoutingAction")
        assign_route_action = etree.SubElement(routing_action, "AssignRouteAction")
        route = etree.SubElement(assign_route_action, "Route", closed="false")


    # Add Trajectories
    for vehicle_name in vehicle_data:
        timestamps = vehicle_data[vehicle_name]['timestamp']
        # Create the trajectory element for this vehicle
        trajectory = etree.SubElement(openscenario, "Trajectory", name=f"{vehicle_name}_Trajectory", closed="false")
        shape = etree.SubElement(trajectory, "Shape")
        polyline = etree.SubElement(shape, "Polyline")

        # Add vertices to the polyline
        for i in range(len(timestamps)):
            vertex = etree.SubElement(polyline, "Vertex", time=str(timestamps[i]))
            position = etree.SubElement(vertex, "Position")
            etree.SubElement(position, "WorldPosition", 
                         x=str(vehicle_data[vehicle_name]['x'][i]), 
                         y=str(vehicle_data[vehicle_name]['y'][i]), 
                         z=str(vehicle_data[vehicle_name]['z'][i]),
                         h=str(vehicle_data[vehicle_name]['yaw'][i]),
                         p=str(vehicle_data[vehicle_name]['pitch'][i]),
                         r=str(vehicle_data[vehicle_name]['roll'][i]),
                         )

    # Convert the XML tree to a string
    xml_str = etree.tostring(openscenario, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    # Write the XML string to a file
    with open(filename, "wb") as file:
        file.write(xml_str)


def compute_velocity(actor_log: pd.DataFrame):
    # Compute differences between successive positions and timestamps
    positions = np.hstack([actor_log[c].to_numpy().reshape((-1,1)) for c in ['x','y','z']])
    timestamps = actor_log['timestamp'].to_numpy()
    delta_positions = np.diff(positions, axis=0)
    delta_timestamps = np.diff(timestamps).reshape((-1, 1))

    # Compute velocities (magnitude of the displacement vector divided by time)
    velocities = delta_positions / delta_timestamps
    # Prepend a zero for the initial velocity and heading (since there's no motion before the first point)
    velocities = np.vstack([[0,0,0], velocities])

    return velocities

def compute_heading(actor_log: pd.DataFrame):
    yaw = np.arctan2(actor_log['y'].to_numpy()[1:].reshape(-1,1), actor_log['x'].to_numpy()[1:].reshape(-1,1))

    # Prepend a zero for the initial velocity and heading (since there's no motion before the first point)
    yaw0 = actor_log['yaw'].to_numpy()[0].reshape(1,1)
    return np.vstack([yaw0, yaw0 + yaw])

def parse_osc(filename: str):
    """
    Parses an OpenSCENARIO file and converts vehicle trajectories to pandas DataFrames with computed motion parameters.

    Parameters:
    - filename: str, the name of the OpenSCENARIO file to parse.

    Returns:
    - vehicle_dataframes: dict, where each key is a vehicle name (str) and each value is a pandas DataFrame containing:
        - timestamps, x, y, z positions, velocities, and headings.
    """
    
    # Parse the OpenSCENARIO file
    tree = etree.parse(filename)
    root = tree.getroot()

    # Initialize the dictionary to store vehicle dataframes
    vehicle_dataframes = []

    # Find all Trajectory elements
    for trajectory in root.findall(".//Trajectory"):
        vehicle_name = trajectory.attrib['name'].replace("_Trajectory", "")
        poses = []
        timestamps = []
        # Extract positions and timestamps
        for vertex in trajectory.findall(".//Vertex"):
            time = float(vertex.attrib['time'])
            position = vertex.find(".//WorldPosition")
            x = float(position.attrib['x'])
            y = float(position.attrib['y'])
            z = float(position.attrib['z'])
            yaw = float(position.attrib.get('h',0.))
            pitch = float(position.attrib.get('p',0.))
            roll = float(position.attrib.get('r',0.))

            timestamps.append(time)
            poses.append([x, y, z, yaw, pitch, roll])
        poses = np.array(poses)
        # Create a DataFrame for the vehicle
        df = {
            'timestamp': timestamps,
            'x': poses[:, 0].tolist(),
            'y': poses[:, 1].tolist(),
            'z': poses[:, 2].tolist(),
            'yaw': poses[:, 3].tolist(),
            'pitch': poses[:, 4].tolist(),
            'roll': poses[:, 5].tolist(),
        }
        df = pd.DataFrame(df)
        df['name'] = [vehicle_name] * len(df['timestamp'])

        # Store the DataFrame in the dictionary
        vehicle_dataframes += [df]

    return vehicle_dataframes

if __name__ == '__main__':


    # Example usage with multiple vehicles
    vehicle1 = {
            'timestamp': [0.0, 2.0, 4.0],
            'x': [0.0, 10.0, 20.0],
            'y': [0.0, 10.0, 15.0],
            'z': [0.0, 0.0, 0.0],
            'yaw': [0.0, 0.0, 0.0],
            'pitch': [0.0, 0.0, 0.0],
            'roll': [0.0, 0.0, 0.0],
            }
    df1 = pd.DataFrame(vehicle1)
    df1['name'] = ['vehicle1'] * len(df1['timestamp'])
    vehicle2 = {
            'timestamp': [0.0, 3.0, 6.0],
            'x': [5.0, 15.0, 25.0],
            'y': [5.0, 10.0, 20.0],
            'z': [0.0, 0.0, 0.0],
            'yaw': [0.0, 0.0, 0.0],
            'pitch': [0.0, 0.0, 0.0],
            'roll': [0.0, 0.0, 0.0],
    }
    df2 = pd.DataFrame(vehicle2)
    df2['name'] = ['vehicle2'] * len(df2['timestamp'])
    df2['yaw'] = compute_heading(df2)
    dfs = [df1, df2]
    for df in dfs:
        v = compute_velocity(df)
        df['vx'] = v[:, 0]
        df['vy'] = v[:, 1]
        df['yaw'] = compute_heading(df)

    print( pd.concat(dfs).to_string())
    
    generate_osc({'vehicle1':df1, 'vehicle2':df2}, "multiple_vehicles_trajectory.xosc")
    dfs_parsed = parse_osc("multiple_vehicles_trajectory.xosc")
    for df in dfs_parsed:
        v = compute_velocity(df)
        df['vx'] = v[:, 0]
        df['vy'] = v[:, 1]
    print(pd.concat(dfs_parsed).to_string())
    diff = pd.concat(dfs).compare(pd.concat(dfs_parsed))

    print('Difference: ')
    print(diff)
    assert pd.concat(dfs).equals(pd.concat(dfs_parsed))