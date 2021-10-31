from collections import *
import xml.etree.ElementTree as ET
import torch
import traci


class SumoController:
    def __init__(self, name: str):
        self.name = name
        self.net_xml = self.name + ".net.xml"
        self.rou_xml = self.name + ".rou.xml"
        self.sumocfg = self.name + ".sumocfg"
        self._reset_tl_logic()

        self.num_junctions = None  # total number of junctions with traffic lights
        self.junctions_id = []
        self.approaching_lanes_id_in_junction = {}  # {junction_id -> its approaching lanes}
        self.last_step_vehicles_at_lane = {}  # {lane_id -> vehicles_id}
        self._init_approaching_lanes_in_junction()

        self.num_phases_in_junction = {}  # {junction_id -> total number of traffic light phases}
        self._init_num_phases_in_junction()

        self.vehicle_enter_time = {}  # {vehicle_id -> its entering time}

        # self._generate_synthetic()
        self._value_check()

    def _init_approaching_lanes_in_junction(self):
        tree = ET.parse(self.net_xml)
        root = tree.getroot()
        connections = root.findall('connection')

        # Find all junctions with traffic lights
        for junction in root.iter(tag="tlLogic"):
            junction_id = junction.attrib['id']
            self.junctions_id.append(junction_id)
            self.approaching_lanes_id_in_junction[junction_id] = []

            # Construct connections for junction and lanes
            for connection in connections:
                if 'tl' in connection.keys() and connection.attrib['tl'] == "{}".format(junction_id):
                    lane_id = "{}_{}".format(connection.attrib['from'], connection.attrib['fromLane'])
                    self.approaching_lanes_id_in_junction[junction_id].append(lane_id)

                    # Initialize vehicles at lane
                    self.last_step_vehicles_at_lane[lane_id] = []

            self.approaching_lanes_id_in_junction[junction_id] = \
                list(set(self.approaching_lanes_id_in_junction[junction_id]))

        self.num_junctions = len(self.junctions_id)

    def _init_num_phases_in_junction(self):
        tree = ET.parse(self.net_xml)
        root = tree.getroot()

        for junction in root.iter(tag="tlLogic"):
            junction_id = junction.attrib['id']
            num_phases = len(junction.findall("phase"))
            self.num_phases_in_junction[junction_id] = num_phases

    def set_tls_to_next_phase_at_junction(self, junction_id):
        current_phase = self.get_tls_current_phase_at_junction(junction_id)
        next_phase = (current_phase + 1) % self.num_phases_in_junction[junction_id]
        traci.trafficlight.setPhase(junction_id, next_phase)

    def get_tls_current_phase_at_junction(self, junction_id):
        # Actually, no need to use mod(%) ...
        return traci.trafficlight.getPhase(junction_id) % self.num_phases_in_junction[junction_id]

    def get_tls_next_phase_at_junction(self, junction_id):
        return (traci.trafficlight.getPhase(junction_id) + 1) % self.num_phases_in_junction[junction_id]

    def get_queue_length_at_junction(self, junction_id):
        lanes_id = self.approaching_lanes_id_in_junction[junction_id]
        total_queue_length = 0
        for lane_id in lanes_id:
            total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
        return total_queue_length

    def get_vehicle_number_at_junction(self, junction_id):
        lanes_id = self.approaching_lanes_id_in_junction[junction_id]
        total_vehicle_number = 0
        for lane_id in lanes_id:
            total_vehicle_number += traci.lane.getLastStepVehicleNumber(lane_id)
        return total_vehicle_number

    def get_updated_waiting_time_at_junction(self, junction_id):
        lanes_id = self.approaching_lanes_id_in_junction[junction_id]
        total_updated_waiting_time = 0
        for lane_id in lanes_id:
            total_updated_waiting_time += traci.lane.getWaitingTime(lane_id)
        return total_updated_waiting_time / 60.0

    def get_delay_at_junction(self, junction_id):
        lanes_id = self.approaching_lanes_id_in_junction[junction_id]
        total_delay = 0
        for lane_id in lanes_id:
            total_delay += 1. - traci.lane.getLastStepMeanSpeed(lane_id) / traci.lane.getMaxSpeed(lane_id)
        return total_delay

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def get_passed_vehicle_number_and_travel_time_at_junction(self, junction_id):
        """
        Can ONLY call once each time step

        :param junction_id:
        :return:
        """
        lanes_id = self.approaching_lanes_id_in_junction[junction_id]
        total_passed_vehicle_number = 0
        total_passed_vehicle_travel_time = 0
        for lane_id in lanes_id:
            current_step_vehicles_id_at_lane = traci.lane.getLastStepVehicleIDs(lane_id)
            passed_vehicles_id = list(set(self.last_step_vehicles_at_lane[lane_id])
                                      - set(current_step_vehicles_id_at_lane))
            # passed vehicle number
            passed_vehicles_number = len(passed_vehicles_id)
            if passed_vehicles_number < 0:
                raise ValueError("Passed vehicle number cannot be < 0")
            total_passed_vehicle_number += passed_vehicles_number

            # passed vehicle travel time
            for vehicle_id in passed_vehicles_id:
                total_passed_vehicle_travel_time += self.get_current_time() - self.vehicle_enter_time[vehicle_id]

            """ VERY IMPORTANT: Reset """
            self.last_step_vehicles_at_lane[lane_id] = current_step_vehicles_id_at_lane
        return total_passed_vehicle_number, total_passed_vehicle_travel_time / 60.0

    def get_current_time(self):
        return traci.simulation.getTime()

    def update_vehicle_enter_time(self):
        vehicles_id = traci.vehicle.getIDList()
        for vehicle_id in vehicles_id:
            if vehicle_id not in self.vehicle_enter_time.keys():
                self.vehicle_enter_time[vehicle_id] = self.get_current_time()

    def get_vehicle_number_in_the_network(self):
        return len(traci.vehicle.getIDList())

    def _generate_synthetic(self):
        # routes
        root = ET.Element("routes")

        # vType
        v_type = ET.SubElement(root, "vType")
        v_type.set("id", "car1")
        v_type.set("type", "passenger")
        v_type.set("length", "5")
        v_type.set("accel", "3.5")
        v_type.set("decel", "2.2")
        v_type.set("sigma", "1.0")
        v_type.set("maxspeed", "70")

        # flow1
        flow1 = ET.SubElement(root, "flow")
        flow1.set("id", "flow1")
        flow1.set("type", "car1")
        flow1.set("beg", "0")
        flow1.set("end", "600")
        flow1.set("number", "120")
        flow1.set("from", "gneE29")
        flow1.set("to", "gneE16")

        # flow2
        flow2 = ET.SubElement(root, "flow")
        flow2.set("id", "flow2")
        flow2.set("type", "car1")
        flow2.set("beg", "0")
        flow2.set("end", "600")
        flow2.set("number", "80")
        flow2.set("from", "gneE15")
        flow2.set("to", "gneE27")

        # flow3
        flow3 = ET.SubElement(root, "flow")
        flow3.set("id", "flow3")
        flow3.set("type", "car1")
        flow3.set("beg", "0")
        flow3.set("end", "600")
        flow3.set("number", "40")
        flow3.set("from", "gneE22")
        flow3.set("to", "gneE8")

        # flow4
        flow4 = ET.SubElement(root, "flow")
        flow4.set("id", "flow4")
        flow4.set("type", "car1")
        flow4.set("beg", "0")
        flow4.set("end", "600")
        flow4.set("number", "15")
        flow4.set("from", "gneE6")
        flow4.set("to", "gneE24")

        # xml file
        trees = ET.ElementTree(root)
        self._pretty_xml(root)
        trees.write(self.rou_xml, encoding="UTF-8")

    def _reset_tl_logic(self):
        tree = ET.parse(self.net_xml)
        root = tree.getroot()
        for tl_logic in root.iter(tag="tlLogic"):
            tl_logic.set("type", "static")
            for phase in tl_logic.iter("phase"):
                # duration = phase.attrib["duration"]
                state = phase.attrib["state"]
                phase.clear()
                phase.set("duration", "10000")
                phase.set("state", state)

        self._pretty_xml(root)
        tree.write(self.net_xml, encoding="UTF-8")

    def _pretty_xml(self, element, indent='\t', newline='\n', level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
        if element:  # 判断element是否有子元素
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
                subelement.tail = newline + indent * level
            self._pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作

    def _value_check(self):
        if self.num_junctions is None:
            raise AttributeError("num_junctions is None")

    def test(self):
        tree = ET.parse(self.net_xml)
        root = tree.getroot()
        print(root.tag, root.attrib)
        for child in root.iter("tlLogic"):
            print(child.tag, child.attrib)
            for c in child:
                print(c.tag, c.attrib, c.text)

    # @staticmethod
    # def _get_all_vehicles_id():
    #     return traci.vehicle.getIDList()
    #
    # @staticmethod
    # def _get_queue_length_at_lane(lane_id):
    #     return traci.lane.getLastStepHaltingNumber(lane_id)
    #
    # @staticmethod
    # def _get_vehicle_number_at_lane(lane_id):
    #     return traci.lane.getLastStepVehicleNumber(lane_id)
    #
    # @staticmethod
    # def _get_updated_waiting_time_of_vehicle_at_lane(lane_id):
    #     return traci.lane.getWaitingTime(lane_id)
    #
    # @staticmethod
    # def _get_delay_at_lane(lane_id):
    #     return 1 - traci.lane.getLastStepMeanSpeed(lane_id) / traci.lane.getMaxSpeed(lane_id)
    #
    # @staticmethod
    # def _get_vehicles_id_at_lane(lane_id):
    #     return traci.lane.getLastStepVehicleIDs(lane_id)
