import tkinter as tk
from tkinter import messagebox, ttk
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import time

# Floors: B2, B1, G, 2–8
FLOORS = [-2, -1, 0, 2, 3, 4, 5, 6, 7, 8]
SHAFTS = 6
CANVAS_W = 600
CANVAS_H = 600
BUILD_L = 60
BUILD_R = CANVAS_W - 60
BUILD_T = 20
BUILD_B = CANVAS_H - 20
FLOOR_COUNT = len(FLOORS)
FLOOR_H = (BUILD_B - BUILD_T) / (FLOOR_COUNT - 1)
CAR_WIDTH = 30

# Entity Colors for Staff (red), Teachers (blue), Students (green)
ENTITY_COLORS = {
    "staff": "red",
    "teacher": "blue",
    "student": "green"
}


class Elevator:
    def __init__(self, canvas, name, served_floors, shaft_idx, app, entity_type):
        self.canvas = canvas
        self.app = app
        self.name = name
        self.served_floors = served_floors
        self.current_floor = 0
        self.prev_floor = 0
        self.stops = []
        self.wait_times = []
        self.service_times = []
        self.busy_steps = 0
        self.entity_type = entity_type
        self.people_in_elevator = []
        self.queue = []
        self.shaft_idx = shaft_idx
        self.wait_time_at_floor = 0
        self.max_wait_time = random.randint(3, 8)

        # Analytics data
        self.trip_history = []  # (origin, destination, passengers, travel_time)
        self.wait_time_by_entity = defaultdict(list)
        self.service_time_by_entity = defaultdict(list)
        self.queue_length_history = []
        self.utilization_history = []
        self.load_factor_history = []
        self.energy_consumption = 0
        self.total_trips = 0
        self.total_passengers_served = 0

        spacing = (BUILD_R - BUILD_L) / (SHAFTS - 1)
        self.x = BUILD_L + shaft_idx * spacing

        # Only create visual elements if not in batch mode
        self.car_id = None
        self.label_id = None
        if not app.batch_mode:
            self.create_elevator_car()
            self.label_id = canvas.create_text(self.x, BUILD_B + 10, text=self.name, anchor='n')

    def create_elevator_car(self):
        if self.app.batch_mode:
            return
        y = self.floor_to_y(self.current_floor)
        self.car_id = self.canvas.create_rectangle(
            self.x - CAR_WIDTH / 2, y - FLOOR_H / 3,
            self.x + CAR_WIDTH / 2, y + FLOOR_H / 3,
            fill='skyblue', outline='black'
        )

    def floor_to_y(self, floor):
        idx = FLOORS.index(floor)
        return BUILD_B - idx * FLOOR_H

    def reset(self):
        self.current_floor = 0
        self.prev_floor = 0
        self.stops.clear()
        self.wait_times.clear()
        self.service_times.clear()
        self.busy_steps = 0
        self.people_in_elevator.clear()
        self.queue.clear()
        self.wait_time_at_floor = 0
        self.max_wait_time = random.randint(3, 8)
        self.trip_history.clear()
        self.wait_time_by_entity.clear()
        self.service_time_by_entity.clear()
        self.queue_length_history.clear()
        self.utilization_history.clear()
        self.load_factor_history.clear()
        self.energy_consumption = 0
        self.total_trips = 0
        self.total_passengers_served = 0

        if not self.app.batch_mode:
            if self.car_id:
                self.canvas.delete(self.car_id)
            if self.label_id:
                self.canvas.delete(self.label_id)
            self.canvas.delete(f"people_{self.name}")
            self.canvas.delete(f"queue_{self.name}")
            self.create_elevator_car()
            self.label_id = self.canvas.create_text(self.x, BUILD_B + 10, text=self.name, anchor='n')

    def step(self):
        self.queue_length_history.append(len(self.queue))
        self.handle_passenger_alighting()
        self.try_board_passengers()

        busy = bool(self.people_in_elevator or self.stops)
        self.utilization_history.append(1 if busy else 0)
        self.load_factor_history.append(len(self.people_in_elevator) / 15.0)

        if self.should_move():
            self.move_elevator()
        else:
            self.wait_time_at_floor += 1

    def handle_passenger_alighting(self):
        to_remove = []
        for i, (entity, dest, board_time) in enumerate(self.people_in_elevator):
            if dest == self.current_floor:
                stime = self.app.sim_time - board_time
                self.service_times.append(stime)
                self.service_time_by_entity[entity].append(stime)
                self.total_passengers_served += 1
                to_remove.append(i)
        for i in reversed(to_remove):
            self.people_in_elevator.pop(i)

        # Only draw if not in batch mode
        if not self.app.batch_mode:
            self.draw_people_in_elevator()

    def try_board_passengers(self):
        if len(self.people_in_elevator) < 15 and self.queue:
            n = min(15, 15 - len(self.people_in_elevator), len(self.queue))
            for _ in range(n):
                entity, join_t = self.queue.pop(0)
                wtime = self.app.sim_time - join_t
                self.wait_times.append(wtime)
                self.wait_time_by_entity[entity].append(wtime)
                dests = [f for f in self.served_floors if f != self.current_floor]
                if dests:
                    dest = random.choice(dests)
                    self.people_in_elevator.append((entity, dest, self.app.sim_time))

        if random.random() < self.get_queue_probability():
            self.queue.append((self.assign_entity_color(), self.app.sim_time))
        if len(self.queue) > 10:
            self.queue = self.queue[-10:]

        # Only draw if not in batch mode
        if not self.app.batch_mode:
            self.draw_people_in_elevator()
            self.draw_queue()

    def get_queue_probability(self):
        if self.name == "E6": return 0.03  # Low probability for E6
        if self.name in ["E1", "E2"]: return 0.08  # Slightly higher probability for E1 and E2
        if self.name in ["E3", "E4", "E5"]: return 0.15  # Increase probability for E3, E4, E5
        return 0.25  # Default probability for other elevators

    def should_move(self):
        if not self.stops: return False
        cap = len(self.people_in_elevator)

        # Adjust the conditions for elevators E3, E4, E5
        if self.name in ["E3", "E4", "E5"]:
            if cap >= 3 or self.wait_time_at_floor >= 4:  # Reduce the requirement for E3 to E5
                self.wait_time_at_floor = 0
                self.max_wait_time = random.randint(3, 6)
                return True

        # For other elevators, use the default conditions
        if self.name == "E6":
            if cap >= 3 or self.wait_time_at_floor >= 4:
                self.wait_time_at_floor = 0
                self.max_wait_time = random.randint(3, 6)
                return True
        elif self.name in ["E1", "E2"]:
            if cap >= 6 or self.wait_time_at_floor >= 6:
                self.wait_time_at_floor = 0
                self.max_wait_time = random.randint(4, 8)
                return True
        else:
            if cap >= 10 or self.wait_time_at_floor >= self.max_wait_time:
                self.wait_time_at_floor = 0
                self.max_wait_time = random.randint(3, 8)
                return True
        return False

    def move_elevator(self):
        if not self.stops: return
        nxt = self.stops[0]
        target = nxt[0] if isinstance(nxt, tuple) else nxt
        ci, ti = FLOORS.index(self.current_floor), FLOORS.index(target)
        ni = ci + 1 if ti > ci else ci - 1 if ti < ci else ci
        origin = self.current_floor
        self.prev_floor = origin
        self.current_floor = FLOORS[ni]
        if self.current_floor != origin:
            self.busy_steps += 1
            self.energy_consumption += abs(ni - ci)

        # Only update visual position if not in batch mode
        if not self.app.batch_mode:
            y = self.floor_to_y(self.current_floor)
            try:
                if self.car_id:
                    x0, y0, x1, y1 = self.canvas.coords(self.car_id)
                    cy = (y0 + y1) / 2
                    self.canvas.move(self.car_id, 0, y - cy)
            except:
                self.create_elevator_car()

        if self.current_floor == target:
            stop = self.stops.pop(0)
            if isinstance(stop, tuple):
                orig, req_t, dest = stop
                trav = self.app.sim_time - req_t
                cnt = len(self.people_in_elevator)
                self.trip_history.append((orig, dest, cnt, trav))
                self.total_trips += 1
                self.stops.insert(0, dest)

    def draw_people_in_elevator(self):
        if self.app.batch_mode:
            return
        self.canvas.delete(f"people_{self.name}")
        y = self.floor_to_y(self.current_floor)
        for i, (entity, _, _) in enumerate(self.people_in_elevator):
            color = ENTITY_COLORS[entity]
            px = self.x + 15 + (i % 3) * 8
            py = y - 10 + (i // 3) * 8
            self.canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill=color, tags=f"people_{self.name}")

    def assign_entity_color(self):
        if self.name in ["E1", "E2"]:
            return random.choices(["staff", "teacher"], weights=[30, 70])[0]
        if self.name == "E6": return "staff"
        return random.choices(["student", "staff", "teacher"], weights=[90, 3, 7])[0]

    def draw_queue(self):
        if self.app.batch_mode:
            return
        self.canvas.delete(f"queue_{self.name}")
        qy = BUILD_B + 30 + (self.shaft_idx * 15)
        for i, (entity, _) in enumerate(self.queue):
            color = ENTITY_COLORS[entity]
            qx = BUILD_L + i * 25
            self.canvas.create_oval(qx - 5, qy - 5, qx + 5, qy + 5, fill=color, tags=f"queue_{self.name}")


class AnalyticsWindow:
    def __init__(self, parent, elevators, sim_data):
        self.window = tk.Toplevel(parent)
        self.window.title("Elevator Analytics Dashboard")
        self.window.geometry("1400x900")
        self.elevators = elevators
        self.sim_data = sim_data
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # add all analysis tabs
        self.create_wait_time_analysis()
        self.create_utilization_analysis()
        self.create_queue_analysis()
        self.create_throughput_analysis()
        self.create_efficiency_analysis()
        self.create_load_factor_analysis()
        self.create_demand_supply_analysis()
        self.create_service_time_analysis()
        self.create_energy_analysis()
        self.create_movement_analysis()

    def create_wait_time_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Wait Time Distribution")
        fig = Figure(figsize=(14, 8))

        # 1) Wait time distribution by entity type
        ax1 = fig.add_subplot(221)
        wait_times_by_entity = {'staff': [], 'teacher': [], 'student': []}
        for elev in self.elevators:
            for entity, times in elev.wait_time_by_entity.items():
                wait_times_by_entity[entity].extend(times)
        for entity, times in wait_times_by_entity.items():
            if times:
                ax1.hist(
                    times,
                    bins=20,
                    alpha=0.7,
                    label=f"{entity.capitalize()} (n={len(times)})",
                    color=ENTITY_COLORS[entity]
                )
        ax1.set_xlabel("Wait Time (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Wait Time Distribution by Entity Type")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) Wait time distribution by elevator (boxplot)
        ax2 = fig.add_subplot(222)
        all_waits, labels = [], []
        for elev in self.elevators:
            if elev.wait_times:
                all_waits.append(elev.wait_times)
                labels.append(elev.name)
        if all_waits:
            ax2.boxplot(all_waits, labels=labels)
            ax2.set_xlabel("Elevator")
            ax2.set_ylabel("Wait Time (seconds)")
            ax2.set_title("Wait Time by Elevator")
            ax2.grid(True, alpha=0.3)

        # 3) Average wait time over sim time
        ax3 = fig.add_subplot(223)
        if self.sim_data['times']:
            ax3.plot(self.sim_data['times'], self.sim_data['waits'])
            ax3.set_xlabel("Simulation Time (s)")
            ax3.set_ylabel("Avg Wait (s)")
            ax3.set_title("Average Wait Time Over Time")
            ax3.grid(True, alpha=0.3)

        # 4) Summary table
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        stats = []
        for entity, times in wait_times_by_entity.items():
            if times:
                stats.append([
                    entity.capitalize(),
                    len(times),
                    f"{np.mean(times):.2f}",
                    f"{np.std(times):.2f}",
                    f"{np.min(times):.2f}",
                    f"{np.max(times):.2f}"
                ])
        if stats:
            table = ax4.table(
                cellText=stats,
                colLabels=["Entity", "Count", "Mean", "Std Dev", "Min", "Max"],
                cellLoc="center",
                loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
        ax4.set_title("Wait Time Stats")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_utilization_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Elevator Utilization")
        fig = Figure(figsize=(14, 8))

        # 1) Utilization over time (MA)
        ax1 = fig.add_subplot(221)
        for elev in self.elevators:
            hist = elev.utilization_history
            if hist and len(hist) > 1:
                w = min(50, len(hist))
                smooth = np.convolve(hist, np.ones(w)/w, mode="valid")
                ax1.plot(range(len(smooth)), smooth, label=elev.name)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Utilization Rate")
        ax1.set_title("Utilization Over Time")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # 2) Avg utilization per elevator
        ax2 = fig.add_subplot(222)
        names, avgs = [], []
        for elev in self.elevators:
            if elev.utilization_history:
                names.append(elev.name)
                avgs.append(np.mean(elev.utilization_history))
        if names:
            bars = ax2.bar(names, avgs, color=['skyblue']*len(names))
            ax2.set_xlabel("Elevator"); ax2.set_ylabel("Avg Util")
            ax2.set_title("Avg Utilization"); ax2.grid(True, alpha=0.3)
            for bar, val in zip(bars, avgs):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center")

        # 3) Busy steps
        ax3 = fig.add_subplot(223)
        names = [e.name for e in self.elevators]
        steps = [e.busy_steps for e in self.elevators]
        bars = ax3.bar(names, steps, color=['lightgreen']*len(names))
        ax3.set_xlabel("Elevator"); ax3.set_ylabel("Busy Steps")
        ax3.set_title("Total Busy Steps"); ax3.grid(True, alpha=0.3)
        for bar, val in zip(bars, steps):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(steps)*0.01, str(val), ha="center")

        # 4) Stats table
        ax4 = fig.add_subplot(224); ax4.axis('off')
        tbl = []
        for elev in self.elevators:
            hist = elev.utilization_history
            if hist:
                arr = np.array(hist)
                tbl.append([
                    elev.name,
                    f"{arr.mean():.3f}",
                    f"{arr.std():.3f}",
                    f"{arr.sum()}",
                    str(elev.busy_steps)
                ])
        if tbl:
            table = ax4.table(
                cellText=tbl,
                colLabels=["Elevator","Mean","Std Dev","Active Ticks","Busy"],
                cellLoc="center", loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1,1.5)
        ax4.set_title("Utilization Stats")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_queue_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Queue Analysis")
        fig = Figure(figsize=(14, 8))

        # 1) Queue over time
        ax1 = fig.add_subplot(221)
        if self.sim_data['times']:
            ax1.plot(self.sim_data['times'], self.sim_data['queues'])
            ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Total Queue")
            ax1.set_title("Queue Length Over Time"); ax1.grid(True, alpha=0.3)

        # 2) Avg queue per elevator
        ax2 = fig.add_subplot(222)
        names, avgs = [], []
        for e in self.elevators:
            if e.queue_length_history:
                names.append(e.name)
                avgs.append(np.mean(e.queue_length_history))
        if names:
            bars = ax2.bar(names, avgs, color=['coral']*len(names))
            ax2.set_xlabel("Elevator"); ax2.set_ylabel("Avg Queue")
            ax2.set_title("Avg Queue Length"); ax2.grid(True, alpha=0.3)
            for bar, val in zip(bars, avgs):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.2f}", ha="center")

        # 3) Distribution
        ax3 = fig.add_subplot(223)
        all_q = [q for e in self.elevators for q in e.queue_length_history]
        if all_q:
            ax3.hist(all_q, bins=20, alpha=0.7)
            ax3.set_xlabel("Queue Length"); ax3.set_ylabel("Freq")
            ax3.set_title("Queue Dist"); ax3.grid(True, alpha=0.3)

        # 4) Table
        ax4 = fig.add_subplot(224); ax4.axis('off')
        tbl = []
        for e in self.elevators:
            if e.queue_length_history:
                arr = np.array(e.queue_length_history)
                tbl.append([e.name, f"{arr.mean():.2f}", f"{arr.std():.2f}", str(arr.min()), str(arr.max())])
        if tbl:
            table = ax4.table(
                cellText=tbl,
                colLabels=["Elevator","Mean","Std","Min","Max"],
                cellLoc="center", loc="center"
            )
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
        ax4.set_title("Queue Stats")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_throughput_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Throughput Analysis")
        fig = Figure(figsize=(14, 8))

        # 1) Cumulative fulfilled
        ax1 = fig.add_subplot(221)
        if self.sim_data['times']:
            ax1.plot(self.sim_data['times'], self.sim_data['fulfilled_requests'])
            ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Cum Req")
            ax1.set_title("Throughput Over Time"); ax1.grid(True, alpha=0.3)

        # 2) Served by elevator
        ax2 = fig.add_subplot(222)
        names = [e.name for e in self.elevators]
        served = [e.total_passengers_served for e in self.elevators]
        bars = ax2.bar(names, served, color=['skyblue']*len(names))
        ax2.set_xlabel("Elevator"); ax2.set_ylabel("Served")
        ax2.set_title("Passengers Served"); ax2.grid(True, alpha=0.3)
        for bar,val in zip(bars,served):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(served)*0.01, str(val), ha="center")

        # 3) Total trips
        ax3 = fig.add_subplot(223)
        trips = [e.total_trips for e in self.elevators]
        bars = ax3.bar(names, trips, color=['lightgreen']*len(names))
        ax3.set_xlabel("Elevator"); ax3.set_ylabel("Trips")
        ax3.set_title("Total Trips"); ax3.grid(True, alpha=0.3)
        for bar,val in zip(bars,trips):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(trips)*0.01, str(val), ha="center")

        # 4) Efficiency (passengers/trip)
        ax4 = fig.add_subplot(224)
        eff = []
        for e in self.elevators:
            eff.append(e.total_passengers_served / e.total_trips if e.total_trips>0 else 0)
        bars = ax4.bar(names, eff, color=['coral']*len(names))
        ax4.set_xlabel("Elevator"); ax4.set_ylabel("Pass/Trip")
        ax4.set_title("Throughput Eff"); ax4.grid(True, alpha=0.3)
        for bar,val in zip(bars,eff):
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.2f}", ha="center")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_efficiency_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="System Efficiency")
        fig = Figure(figsize=(14, 8))

        # 1) Efficiency over time
        ax1 = fig.add_subplot(221)
        if self.sim_data['times'] and self.sim_data['efficiency']:
            ax1.plot(self.sim_data['times'], self.sim_data['efficiency'])
            ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Efficiency")
            ax1.set_title("System Efficiency Over Time"); ax1.grid(True, alpha=0.3)

        # 2) Service rate by elevator
        ax2 = fig.add_subplot(222)
        names = [e.name for e in self.elevators]
        rates = []
        for e in self.elevators:
            if e.service_times:
                mean_t = np.mean(e.service_times)
                rates.append(1/mean_t if mean_t>0 else 0)
            else:
                rates.append(0)
        bars = ax2.bar(names, rates, color=['skyblue']*len(names))
        ax2.set_xlabel("Elevator"); ax2.set_ylabel("Serv/s")
        ax2.set_title("Service Rate by Elevator"); ax2.grid(True, alpha=0.3)
        for bar,val in zip(bars,rates):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(rates)*0.01, f"{val:.3f}", ha="center")

        # 3) Response time dist
        ax3 = fig.add_subplot(223)
        resp = []
        for e in self.elevators:
            resp.extend(e.wait_times + e.service_times)
        if resp:
            ax3.hist(resp, bins=30, alpha=0.7)
            ax3.set_xlabel("Response Time"); ax3.set_ylabel("Freq")
            ax3.set_title("Response Time Dist"); ax3.grid(True, alpha=0.3)

        # 4) Metrics table
        ax4 = fig.add_subplot(224); ax4.axis('off')
        total_pass = sum(e.total_passengers_served for e in self.elevators)
        total_wait = sum(len(e.wait_times) for e in self.elevators)
        total_serv = sum(len(e.service_times) for e in self.elevators)
        avg_util = np.mean([np.mean(e.utilization_history) if e.utilization_history else 0 for e in self.elevators])
        data = [
            ["Total Served", str(total_pass)],
            ["Wait Events", str(total_wait)],
            ["Service Events", str(total_serv)],
            ["Avg System Util", f"{avg_util:.3f}"],
            ["Active Elevators", str(len([e for e in self.elevators if e.total_passengers_served>0]))]
        ]
        table = ax4.table(cellText=data, colLabels=["Metric","Value"], cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
        ax4.set_title("Efficiency Metrics")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_load_factor_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Load Factor Analysis")
        fig = Figure(figsize=(14, 8))

        # 1) Load factor over time
        ax1 = fig.add_subplot(221)
        for e in self.elevators:
            lf = e.load_factor_history
            if lf and len(lf)>1:
                w = min(50, len(lf))
                smooth = np.convolve(lf, np.ones(w)/w, mode='valid')
                ax1.plot(range(len(smooth)), smooth, label=e.name)
        ax1.set_xlabel("Time Steps"); ax1.set_ylabel("Load Factor")
        ax1.set_title("Load Factor Over Time"); ax1.legend(); ax1.grid(True, alpha=0.3)

        # 2) Avg load factor per elevator
        ax2 = fig.add_subplot(222)
        names, avgs = [], []
        for e in self.elevators:
            if e.load_factor_history:
                names.append(e.name)
                avgs.append(np.mean(e.load_factor_history))
        if names:
            bars = ax2.bar(names, avgs, color=['skyblue']*len(names))
            ax2.set_xlabel("Elevator"); ax2.set_ylabel("Avg Load Factor")
            ax2.set_title("Avg Load Factor"); ax2.grid(True, alpha=0.3)
            for bar,val in zip(bars,avgs):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center")

        # 3) Distribution
        ax3 = fig.add_subplot(223)
        all_lf = [lf for e in self.elevators for lf in e.load_factor_history]
        if all_lf:
            ax3.hist(all_lf, bins=20, alpha=0.7)
            ax3.set_xlabel("Load Factor"); ax3.set_ylabel("Freq")
            ax3.set_title("Load Factor Dist"); ax3.grid(True, alpha=0.3)

        # 4) Table
        ax4 = fig.add_subplot(224); ax4.axis('off')
        tbl = []
        for e in self.elevators:
            if e.load_factor_history:
                arr = np.array(e.load_factor_history)
                tbl.append([e.name, f"{arr.mean():.3f}", f"{arr.std():.3f}", f"{arr.max():.3f}", str((arr>0.8).sum())])
        if tbl:
            table = ax4.table(cellText=tbl, colLabels=["Elevator","Mean","Std","Max",">80%"], cellLoc="center", loc="center")
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
        ax4.set_title("Load Factor Stats")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_demand_supply_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Demand vs Supply")
        fig = Figure(figsize=(14, 8))

        # Demand vs supply over time
        ax1 = fig.add_subplot(221)
        if self.sim_data['times']:
            demand = self.sim_data['queues']
            supply = []
            for i in range(len(self.sim_data['times'])):
                cap=0
                for e in self.elevators:
                    if i<len(e.load_factor_history):
                        used=e.load_factor_history[i]*15
                        cap+=15-used
                supply.append(cap)
            ax1.plot(self.sim_data['times'], demand, label="Demand")
            ax1.plot(self.sim_data['times'], supply, label="Supply")
            ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Count")
            ax1.set_title("Demand vs Supply"); ax1.legend(); ax1.grid(True, alpha=0.3)

        # Peak demand by type
        ax2 = fig.add_subplot(222)
        types = {'E6':[ ],'E1-2':[], 'E3-5':[]}
        for e in self.elevators:
            if e.name=="E6": types['E6'].extend(e.queue_length_history)
            elif e.name in ["E1","E2"]: types['E1-2'].extend(e.queue_length_history)
            else: types['E3-5'].extend(e.queue_length_history)
        names, peaks = [], []
        for k,v in types.items():
            if v: names.append(k); peaks.append(max(v))
        if names:
            bars=ax2.bar(names, peaks, color=['red','blue','green'])
            ax2.set_xlabel("Group"); ax2.set_ylabel("Peak Queue")
            ax2.set_title("Peak Demand"); ax2.grid(True, alpha=0.3)
            for bar,val in zip(bars,peaks):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(val), ha="center")

        # Utilization vs demand scatter
        ax3 = fig.add_subplot(223)
        util, q = [], []
        for e in self.elevators:
            m = min(len(e.utilization_history), len(e.queue_length_history))
            util+=e.utilization_history[:m]
            q+=e.queue_length_history[:m]
        if util and q:
            ax3.scatter(q, util, alpha=0.6)
            ax3.set_xlabel("Queue"); ax3.set_ylabel("Util")
            ax3.set_title("Util vs Demand"); ax3.grid(True, alpha=0.3)

        # Supply-demand metrics
        ax4 = fig.add_subplot(224); ax4.axis('off')
        tot_cap = len(self.elevators)*15
        served = sum(e.total_passengers_served for e in self.elevators)
        avg_q = np.mean([np.mean(e.queue_length_history) for e in self.elevators if e.queue_length_history] or [0])
        cap_util = served/(tot_cap*len(self.sim_data['times'])) if self.sim_data['times'] else 0
        data = [
            ["System Capacity", str(tot_cap)],
            ["Total Served", str(served)],
            ["Avg Queue", f"{avg_q:.2f}"],
            ["Capacity Util", f"{cap_util:.3f}"],
            ["Demand/Supply", f"{avg_q/tot_cap:.3f}" if tot_cap else "N/A"]
        ]
        table = ax4.table(cellText=data, colLabels=["Metric","Value"], cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
        ax4.set_title("Supply-Demand Metrics")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_service_time_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Service Time Analysis")
        fig = Figure(figsize=(14, 8))

        ax1 = fig.add_subplot(221)
        for e in self.elevators:
            if e.service_times and len(e.service_times)>1:
                w = min(50,len(e.service_times))
                smooth = np.convolve(e.service_times, np.ones(w)/w, mode="valid")
                ax1.plot(range(len(smooth)), smooth, label=e.name)
        ax1.set_xlabel("Time Steps"); ax1.set_ylabel("Service Time")
        ax1.set_title("Service Time MA")
        # only draw a legend if we've plotted something with a label
        handles, labels = ax1.get_legend_handles_labels()
        if labels:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(222)
        names, avgs = [], []
        for e in self.elevators:
            if e.service_times:
                names.append(e.name)
                avgs.append(np.mean(e.service_times))
        if names:
            bars=ax2.bar(names,avgs,color=['skyblue']*len(names))
            ax2.set_xlabel("Elevator"); ax2.set_ylabel("Avg Service Time")
            ax2.set_title("Avg Service Time"); ax2.grid(True, alpha=0.3)
            for bar,val in zip(bars,avgs):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center")

        ax3 = fig.add_subplot(223)
        all_st=[]
        for e in self.elevators:
            all_st.extend(e.service_times)
        if all_st:
            ax3.hist(all_st,bins=30,alpha=0.7)
            ax3.set_xlabel("Service Time"); ax3.set_ylabel("Freq")
            ax3.set_title("Service Time Dist"); ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(224); ax4.axis('off')
        tbl=[]
        for e in self.elevators:
            if e.service_times:
                arr=np.array(e.service_times)
                tbl.append([e.name, f"{arr.mean():.3f}", f"{arr.std():.3f}", str(arr.min()), str(arr.max())])
        if tbl:
            table=ax4.table(cellText=tbl, colLabels=["Elevator","Mean","Std","Min","Max"], cellLoc="center", loc="center")
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
        ax4.set_title("Service Time Stats")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_energy_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Energy Consumption Analysis")
        fig = Figure(figsize=(14, 8))

        ax1 = fig.add_subplot(221)
        energy=[]
        for e in self.elevators:
            energy.extend(e.utilization_history)
        if energy:
            ax1.plot(range(len(energy)), energy, label="Energy")
            ax1.set_xlabel("Time Steps"); ax1.set_ylabel("Energy Units")
            ax1.set_title("Energy Over Time"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(222)
        names, sums = [], []
        for e in self.elevators:
            names.append(e.name)
            sums.append(np.sum(e.utilization_history))
        bars=ax2.bar(names, sums, color=['skyblue']*len(names))
        ax2.set_xlabel("Elevator"); ax2.set_ylabel("Total Energy")
        ax2.set_title("Energy by Elevator"); ax2.grid(True, alpha=0.3)
        for bar,val in zip(bars,sums):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center")

        ax3 = fig.add_subplot(223)
        if energy:
            ax3.hist(energy,bins=30,alpha=0.7)
            ax3.set_xlabel("Energy"); ax3.set_ylabel("Freq")
            ax3.set_title("Energy Dist"); ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(224); ax4.axis('off')
        tbl=[]
        for e in self.elevators:
            if e.utilization_history:
                arr=np.array(e.utilization_history)
                tbl.append([e.name, f"{arr.mean():.3f}", f"{arr.std():.3f}", str(arr.min()), str(arr.max())])
        if tbl:
            table=ax4.table(cellText=tbl, colLabels=["Elevator","Mean","Std","Min","Max"], cellLoc="center", loc="center")
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
        ax4.set_title("Energy Stats")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)

    def create_movement_analysis(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Movement Patterns")
        fig = Figure(figsize=(14, 8))

        # Heatmap
        ax1 = fig.add_subplot(221)
        floor_usage = np.zeros((len(FLOORS), len(self.elevators)))
        for idx, e in enumerate(self.elevators):
            for orig, dest, _, _ in e.trip_history:
                oi, di = FLOORS.index(orig), FLOORS.index(dest)
                floor_usage[oi, idx] += 1
                floor_usage[di, idx] += 1
        if np.any(floor_usage):
            im = ax1.imshow(floor_usage, cmap='YlOrRd', aspect='auto')
            ax1.set_xlabel("Elevator"); ax1.set_ylabel("Floor")
            ax1.set_title("Floor Usage Heatmap")
            ax1.set_xticks(range(len(self.elevators)))
            ax1.set_xticklabels([e.name for e in self.elevators])
            ax1.set_yticks(range(len(FLOORS)))
            ax1.set_yticklabels([str(f) for f in FLOORS])
            fig.colorbar(im, ax=ax1)
        else:
            ax1.text(0.5,0.5,"No trip data","center")
        # Movement counts
        ax2 = fig.add_subplot(222)
        names = [e.name for e in self.elevators]
        counts = [e.busy_steps for e in self.elevators]
        if any(counts):
            bars = ax2.bar(names, counts, color=['skyblue']*len(names))
            ax2.set_xlabel("Elevator"); ax2.set_ylabel("Total Steps")
            ax2.set_title("Movement Steps"); ax2.grid(True, alpha=0.3)
            for bar,val in zip(bars,counts):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(counts)*0.01, str(val), ha="center")

        # Trip distance dist
        ax3 = fig.add_subplot(223)
        dists = []
        for e in self.elevators:
            for orig, dest, _, _ in e.trip_history:
                dists.append(abs(FLOORS.index(dest)-FLOORS.index(orig)))
        if dists:
            ax3.hist(dists, bins=range(max(dists)+2), alpha=0.7)
            ax3.set_xlabel("Floors Traveled"); ax3.set_ylabel("Freq")
            ax3.set_title("Trip Distance Dist"); ax3.grid(True, alpha=0.3)

        # Metrics table
        ax4 = fig.add_subplot(224); ax4.axis('off')
        total_trips = sum(len(e.trip_history) for e in self.elevators)
        avg_dist = np.mean(dists) if dists else 0
        total_mov = sum(e.busy_steps for e in self.elevators)
        move_eff = total_trips/total_mov if total_mov>0 else 0
        most_active = FLOORS[np.argmax(floor_usage.sum(axis=1))] if np.any(floor_usage) else 'N/A'
        data = [
            ["Total Trips", str(total_trips)],
            ["Avg Trip Distance", f"{avg_dist:.2f}"],
            ["Total Movement", str(total_mov)],
            ["Movement Eff", f"{move_eff:.3f}"],
            ["Most Active Floor", str(most_active)]
        ]
        table = ax4.table(cellText=data, colLabels=["Metric","Value"], cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
        ax4.set_title("Movement Stats")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(fill="both", expand=True)



class ElevatorSimApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Elevator System Simulation")
        self.root.geometry("1000x800")

        # Canvas for building & elevators
        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H, bg='white')
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.draw_building()

        # Controls & chart area
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.setup_ui(ctrl_frame)

        chart_frame = ttk.Frame(self.root)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.setup_charts(chart_frame)

        # Simulation state
        self.running = False
        self.sim_time = 0.0
        self.sim_data = {'times': [], 'waits': [], 'queues': [], 'fulfilled_requests': [], 'efficiency': []}
        self.batch_mode = False  # New: controls visual updates
        self.speed_multiplier = 1  # New: controls simulation speed
        self.update_interval = 50  # New: controls GUI update frequency

        # Elevators
        self.elevators = self.create_elevators()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self, parent):
        # First row - Basic controls
        ttk.Label(parent, text="Duration (s):").grid(row=0, column=0, padx=5)
        self.duration_var = tk.StringVar(value="3600")
        ttk.Entry(parent, width=6, textvariable=self.duration_var).grid(row=0, column=1)

        ttk.Label(parent, text="λ (/s):").grid(row=0, column=2, padx=5)
        self.lambda_var = tk.StringVar(value="0.1")
        ttk.Entry(parent, width=6, textvariable=self.lambda_var).grid(row=0, column=3)

        ttk.Button(parent, text="Start", command=self.start).grid(row=0, column=4, padx=5)
        ttk.Button(parent, text="Stop", command=self.stop).grid(row=0, column=5, padx=5)
        ttk.Button(parent, text="Reset", command=self.reset).grid(row=0, column=6, padx=5)
        ttk.Button(parent, text="Analytics", command=self.open_analytics).grid(row=0, column=7, padx=5)

        # Second row - Speed controls
        ttk.Label(parent, text="Speed:").grid(row=1, column=0, padx=5)
        self.speed_var = tk.StringVar(value="1")
        speed_combo = ttk.Combobox(parent, width=8, textvariable=self.speed_var,
                                   values=["1", "10", "50", "100", "500", "1000"])
        speed_combo.grid(row=1, column=1, padx=5)
        speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)

        self.batch_var = tk.BooleanVar()
        ttk.Checkbutton(parent, text="Batch Mode (No Animation)",
                        variable=self.batch_var, command=self.on_batch_change).grid(row=1, column=2, columnspan=2,
                                                                                    padx=5)

        # Progress bar
        ttk.Label(parent, text="Progress:").grid(row=1, column=4, padx=5)
        self.progress = ttk.Progressbar(parent, length=200, mode='determinate')
        self.progress.grid(row=1, column=5, columnspan=2, padx=5, sticky='ew')

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var).grid(row=1, column=7, padx=5)

    def on_speed_change(self, event=None):
        try:
            self.speed_multiplier = int(self.speed_var.get())
            # Adjust update interval based on speed
            if self.speed_multiplier >= 100:
                self.update_interval = 200  # Update less frequently for very fast simulations
            elif self.speed_multiplier >= 50:
                self.update_interval = 100
            else:
                self.update_interval = 50
        except ValueError:
            self.speed_multiplier = 1

    def on_batch_change(self):
        self.batch_mode = self.batch_var.get()
        # Reset elevators to apply batch mode
        if hasattr(self, 'elevators'):
            for elevator in self.elevators:
                if self.batch_mode:
                    # Clear visual elements
                    if elevator.car_id:
                        self.canvas.delete(elevator.car_id)
                        elevator.car_id = None
                    if elevator.label_id:
                        self.canvas.delete(elevator.label_id)
                        elevator.label_id = None
                    self.canvas.delete(f"people_{elevator.name}")
                    self.canvas.delete(f"queue_{elevator.name}")
                else:
                    # Recreate visual elements
                    elevator.create_elevator_car()
                    elevator.label_id = self.canvas.create_text(elevator.x, BUILD_B + 10, text=elevator.name,
                                                                anchor='n')

    def setup_charts(self, parent):
        self.fig = Figure(figsize=(6, 8))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.line_wait, = self.ax1.plot([], [], label="Avg Wait")
        self.line_queue, = self.ax1.plot([], [], label="Queue Len")
        self.ax1.legend(loc='upper left', bbox_to_anchor=(.8, 1))
        self.line_eff, = self.ax2.plot([], [], label="Util/Req")
        self.ax2.legend(loc='upper left', bbox_to_anchor=(.8, 1))
        self.line_thr, = self.ax3.plot([], [], label="Fulfilled")
        self.line_den, = self.ax3.plot([], [], label="Demand", linestyle='--')
        self.ax3.legend(loc='upper left', bbox_to_anchor=(.8, 1))
        self.chart = FigureCanvasTkAgg(self.fig, master=parent)
        self.chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_building(self):
        self.canvas.create_rectangle(BUILD_L, BUILD_T, BUILD_R, BUILD_B, outline='black', width=2)
        for i, f in enumerate(FLOORS):
            y = BUILD_B - i * FLOOR_H
            self.canvas.create_line(BUILD_L, y, BUILD_R, y, fill='gray')
            lbl = ("B" + str(-f)) if f < 0 else ("G" if f == 0 else str(f))
            self.canvas.create_text(BUILD_L - 10, y, text=lbl, anchor='e')

    def create_elevators(self):
        groups = (
            FLOORS,
            [f for f in FLOORS if 0 <= f <= 7],
            [f for f in FLOORS if 0 <= f <= 7],
            [f for f in FLOORS if 0 <= f <= 7],
            [f for f in FLOORS if 0 <= f <= 7],
            [f for f in FLOORS if f != 8]
        )
        elevators = []
        for i, grp in enumerate(groups):
            etype = "staff" if i < 2 else "student" if i < 5 else "staff"
            elevators.append(Elevator(self.canvas, f"E{i + 1}", grp, i, self, etype))
        return elevators

    def start(self):
        try:
            self.duration = float(self.duration_var.get())
            self.lam = float(self.lambda_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Enter numeric values.")
            return

        self.sim_time = 0.0
        self.sim_data = {'times': [], 'waits': [], 'queues': [], 'fulfilled_requests': [], 'efficiency': []}
        for e in self.elevators:
            e.reset()

        self.running = True
        self.progress['maximum'] = self.duration
        self.status_var.set(f"Running (Speed: {self.speed_multiplier}x)")

        # Start timing
        self.start_time = time.time()

        self.root.after(10, self.tick)  # Start immediately

    def stop(self):
        self.running = False
        self.status_var.set("Stopped")

    def reset(self):
        self.stop()
        self.sim_time = 0.0
        self.sim_data = {'times': [], 'waits': [], 'queues': [], 'fulfilled_requests': [], 'efficiency': []}
        for e in self.elevators:
            e.reset()

        # Clear and redraw charts
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
        self.setup_charts(self.chart.get_tk_widget().master)

        # Redraw building if not in batch mode
        if not self.batch_mode:
            self.canvas.delete("all")
            self.draw_building()
            for elevator in self.elevators:
                elevator.create_elevator_car()
                elevator.label_id = self.canvas.create_text(elevator.x, BUILD_B + 10, text=elevator.name, anchor='n')

        self.progress['value'] = 0
        self.status_var.set("Ready")

    def tick(self):
        if not self.running:
            return

        # Run multiple simulation steps based on speed multiplier
        steps_to_run = min(self.speed_multiplier, int(self.duration - self.sim_time))

        for _ in range(steps_to_run):
            if self.sim_time >= self.duration:
                break

            self.sim_time += 1

            # Generate requests
            n = np.random.poisson(self.lam)
            for _ in range(n):
                origin = random.choice(FLOORS)
                choices = [e for e in self.elevators if origin in e.served_floors]
                if choices:
                    elev = random.choice(choices)
                    dests = [f for f in elev.served_floors if f != origin]
                    if dests:
                        elev.stops.append((origin, self.sim_time, random.choice(dests)))

            # Step all elevators
            for e in self.elevators:
                e.step()

            # Collect data (sample every 10 steps to reduce memory usage)
            if self.sim_time % 10 == 0:
                waits = [w for e in self.elevators for w in e.wait_times]
                queues = sum(len(e.queue) for e in self.elevators)
                util = np.mean([np.mean(e.utilization_history) if e.utilization_history else 0 for e in self.elevators])
                fulfilled = sum(len(e.wait_times) for e in self.elevators)

                self.sim_data['times'].append(self.sim_time)
                self.sim_data['waits'].append(np.mean(waits) if waits else 0)
                self.sim_data['queues'].append(queues)
                self.sim_data['fulfilled_requests'].append(fulfilled)
                self.sim_data['efficiency'].append(util)

        # Update progress
        self.progress['value'] = self.sim_time

        # Update status with estimated time remaining
        if self.sim_time > 0:
            elapsed_real_time = time.time() - self.start_time
            estimated_total_time = elapsed_real_time * (self.duration / self.sim_time)
            remaining_time = estimated_total_time - elapsed_real_time
            self.status_var.set(f"Running - {self.sim_time:.0f}/{self.duration:.0f}s (ETA: {remaining_time:.1f}s)")

        # Update charts less frequently in batch mode or high speed
        should_update_charts = (
                (not self.batch_mode) or
                (self.sim_time % (self.speed_multiplier * 10) == 0) or
                (self.sim_time >= self.duration)
        )

        if should_update_charts and self.sim_data['times']:
            # Update charts
            t = self.sim_data['times']
            self.line_wait.set_data(t, self.sim_data['waits'])
            self.line_queue.set_data(t, self.sim_data['queues'])
            self.line_eff.set_data(t, self.sim_data['efficiency'])
            self.line_thr.set_data(t, self.sim_data['fulfilled_requests'])
            self.line_den.set_data(t, self.sim_data['queues'])

            for ax in (self.ax1, self.ax2, self.ax3):
                ax.relim()
                ax.autoscale_view()
            self.chart.draw()

        # Continue or finish
        if self.sim_time < self.duration and self.running:
            self.root.after(self.update_interval, self.tick)
        else:
            self.running = False
            elapsed_time = time.time() - self.start_time
            self.status_var.set(f"Completed in {elapsed_time:.1f}s")
            messagebox.showinfo("Done",
                                f"Simulation finished!\nSimulated: {self.sim_time:.0f}s\nReal time: {elapsed_time:.1f}s\nSpeedup: {self.sim_time / elapsed_time:.1f}x")

    def open_analytics(self):
        AnalyticsWindow(self.root, self.elevators, self.sim_data)

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    app = ElevatorSimApp()
    app.root.mainloop()