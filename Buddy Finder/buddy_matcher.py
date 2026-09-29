import os, random
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import customtkinter as ctk
from PIL import Image


INTERESTS = ["gardening", "astronomy", "chess", "music", "coding", "cooking",
             "dance", "robotics", "reading", "3D printing", "yoga", "board games"]
DEPTS = ["Structural", "Operations", "Human Eng", "Automation", "Medical", "Education"]
MODULES = ["Ridge A", "Ridge B", "Crater View", "Sunwell"]

ACCENT = "#d6a35a"
ACCENT_HOVER = "#b98a47"
MUTED = "#8c9db0"
TITLES = ["Welcome", "About you", "Your interests", "Your buddies"]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buddy_matches.png")


# ---------------- matching logic ----------------

def make_person(rng, name):
    return {
        "name": name,
        "age": rng.randint(19, 62),
        "dept": rng.choice(DEPTS),
        "module": rng.choice(MODULES),
        "interests": rng.sample(INTERESTS, rng.randint(3, 5)),
        "years": round(rng.uniform(0.2, 6), 1),
    }

def score(a, b):
    shared = set(a["interests"]) & set(b["interests"])
    union = set(a["interests"]) | set(b["interests"])
    interest_score = len(shared) / len(union) if union else 0
    same_module = a["module"] == b["module"]
    age_score = max(0, 1 - abs(a["age"] - b["age"]) / 40)
    tenure_score = min(1, b["years"] / 5)
    diff_dept = a["dept"] != b["dept"]
    total = 0.45*interest_score + 0.2*same_module + 0.15*age_score + 0.1*tenure_score + 0.1*diff_dept
    return total, shared, same_module

def explain(b, shared, same_module):
    bits = []
    if shared:
        bits.append(f"you both enjoy {', '.join(list(shared)[:2])}")
    if same_module:
        bits.append(f"you're both in {b['module']}")
    bits.append(f"they've been here {b['years']} years")
    return "Matched because " + "; ".join(bits) + "."

def best_matches(newcomer, people, n=3):
    scored = [(p, *score(newcomer, p)) for p in people if p is not newcomer]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]

def draw_graph(pairs, path):
    G = nx.Graph()
    newcomer_names = {p["name"] for p, _ in pairs}
    for newcomer, matches in pairs:
        for buddy, s, *_ in matches:
            G.add_edge(newcomer["name"], buddy["name"], weight=s)
    pos = nx.spring_layout(G, k=2.0, weight=None, iterations=100, seed=1)
    colors = [ACCENT if node in newcomer_names else MUTED for node in G.nodes]
    widths = [(G[u][v].get("weight") or 1.0) * 6 for u, v in G.edges]
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color=colors, width=widths,
            font_size=8, node_size=1000, edge_color="#888")
    plt.margins(0.15)
    plt.title("Buddy matches")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------- window + shared state ----------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Colony Newcomer Survey")
root.geometry("720x800")
root.minsize(640, 700)

state = {
    "step": 0,
    "population": [],
    "pop_size": None,
    "results": [],
    "current": None,
    "graph_img": None,
}

pop_var = ctk.IntVar(value=40)
name_var = ctk.StringVar()
age_var = ctk.IntVar(value=25)
dept_var = ctk.StringVar(value=DEPTS[0])
module_var = ctk.StringVar(value=MODULES[0])
interest_vars = {i: ctk.BooleanVar(value=False) for i in INTERESTS}

# header
header = ctk.CTkFrame(root, fg_color="transparent")
header.pack(fill="x", padx=24, pady=(20, 0))
step_label = ctk.CTkLabel(header, text="", font=ctk.CTkFont(size=13))
step_label.pack(anchor="w")
progress = ctk.CTkProgressBar(header, progress_color=ACCENT)
progress.pack(fill="x", pady=(6, 0))

# body
body = ctk.CTkFrame(root, corner_radius=14)
body.pack(fill="both", expand=True, padx=24, pady=16)

# nav bar
nav = ctk.CTkFrame(root, fg_color="transparent")
nav.pack(fill="x", padx=24, pady=(0, 20))
back_btn = ctk.CTkButton(nav, text="Back", width=110, fg_color="gray35", hover_color="gray25")
back_btn.pack(side="left")
error_label = ctk.CTkLabel(nav, text="", text_color="#e06c6c")
error_label.pack(side="left", padx=14)
next_btn = ctk.CTkButton(nav, text="Next", width=160, fg_color=ACCENT,
                         hover_color=ACCENT_HOVER, text_color="black")
next_btn.pack(side="right")


# ---------------- survey actions ----------------

def count_interests():
    return sum(v.get() for v in interest_vars.values())

def run_matching():
    size = pop_var.get()
    if size != state["pop_size"]:
        rng = random.Random(7)
        state["population"] = [make_person(rng, f"Resident {i}") for i in range(size)]
        state["pop_size"] = size
        state["results"] = []

    name = name_var.get().strip()
    taken = {p["name"] for p, _ in state["results"]}
    base, n = name, 2
    while name in taken:
        name = f"{base} ({n})"
        n += 1

    person = {
        "name": name,
        "age": age_var.get(),
        "dept": dept_var.get(),
        "module": module_var.get(),
        "interests": [i for i, v in interest_vars.items() if v.get()],
        "years": 0.0,
    }
    matches = best_matches(person, state["population"])
    state["results"].append((person, matches))
    state["current"] = (person, matches)
    draw_graph(state["results"], OUT_PATH)

def reset_form():
    name_var.set("")
    for v in interest_vars.values():
        v.set(False)

def go_back():
    if state["step"] > 0:
        state["step"] -= 1
        show()

def go_next():
    step = state["step"]
    if step == 1 and not name_var.get().strip():
        error_label.configure(text="Please enter your name.")
        return
    if step == 2:
        if not 3 <= count_interests() <= 5:
            error_label.configure(text="Pick between 3 and 5 interests.")
            return
        run_matching()
    if step == 3:
        reset_form()
        state["step"] = 1
    else:
        state["step"] += 1
    show()

back_btn.configure(command=go_back)
next_btn.configure(command=go_next)


# ---------------- pages ----------------

def page_welcome():
    ctk.CTkLabel(body, text="Welcome to the colony!",
                 font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(60, 12))
    ctk.CTkLabel(body, text="Answer a few quick questions and we'll pair you\n"
                            "with three residents who can help you settle in.",
                 font=ctk.CTkFont(size=15), justify="center").pack(pady=(0, 40))

    box = ctk.CTkFrame(body, corner_radius=12)
    box.pack(padx=60, fill="x")
    ctk.CTkLabel(box, text="Admin: number of current residents",
                 font=ctk.CTkFont(size=13)).pack(anchor="w", padx=16, pady=(14, 0))
    val = ctk.CTkLabel(box, text=str(pop_var.get()), font=ctk.CTkFont(size=18, weight="bold"))
    ctk.CTkSlider(box, from_=10, to=100, number_of_steps=90, variable=pop_var,
                  button_color=ACCENT,
                  command=lambda v: val.configure(text=str(int(v)))).pack(fill="x", padx=16, pady=8)
    val.pack(pady=(0, 14))

def field_label(text):
    ctk.CTkLabel(body, text=text, font=ctk.CTkFont(size=14)).pack(anchor="w", padx=30, pady=(12, 4))

def page_about():
    ctk.CTkLabel(body, text="Tell us about yourself",
                 font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=30, pady=(30, 20))

    field_label("What's your name?")
    ctk.CTkEntry(body, textvariable=name_var, placeholder_text="e.g. Alex",
                 height=36).pack(fill="x", padx=30)

    field_label("How old are you?")
    age_row = ctk.CTkFrame(body, fg_color="transparent")
    age_row.pack(fill="x", padx=30)
    age_lbl = ctk.CTkLabel(age_row, text=str(age_var.get()), width=40,
                           font=ctk.CTkFont(size=16, weight="bold"))
    ctk.CTkSlider(age_row, from_=18, to=65, number_of_steps=47, variable=age_var,
                  button_color=ACCENT,
                  command=lambda v: age_lbl.configure(text=str(int(v)))).pack(side="left", fill="x", expand=True)
    age_lbl.pack(side="right", padx=(10, 0))

    field_label("Which department are you joining?")
    ctk.CTkOptionMenu(body, values=DEPTS, variable=dept_var, height=34).pack(anchor="w", padx=30)

    field_label("Which module will you live in?")
    ctk.CTkSegmentedButton(body, values=MODULES, variable=module_var,
                           selected_color=ACCENT, selected_hover_color=ACCENT_HOVER).pack(fill="x", padx=30)

def page_interests():
    ctk.CTkLabel(body, text="What do you enjoy?",
                 font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=30, pady=(30, 4))
    ctk.CTkLabel(body, text="Pick 3 to 5 so we can find people who share them.",
                 font=ctk.CTkFont(size=14)).pack(anchor="w", padx=30, pady=(0, 18))

    grid = ctk.CTkFrame(body, fg_color="transparent")
    grid.pack(fill="x", padx=30)
    counter = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=14, weight="bold"))

    def update_counter():
        n = count_interests()
        counter.configure(text=f"{n} selected", text_color=ACCENT if 3 <= n <= 5 else "gray60")

    for i, interest in enumerate(INTERESTS):
        ctk.CTkCheckBox(grid, text=interest, variable=interest_vars[interest],
                        fg_color=ACCENT, hover_color=ACCENT_HOVER,
                        command=update_counter).grid(row=i // 3, column=i % 3, sticky="w", padx=8, pady=10)
    for c in range(3):
        grid.grid_columnconfigure(c, weight=1)

    counter.pack(pady=20)
    update_counter()

def page_results():
    person, matches = state["current"]
    scroll = ctk.CTkScrollableFrame(body, fg_color="transparent")
    scroll.pack(fill="both", expand=True, padx=6, pady=6)

    ctk.CTkLabel(scroll, text=f"Here are your buddies, {person['name']}!",
                 font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=16, pady=(16, 2))
    ctk.CTkLabel(scroll, text=f"{person['dept']} | {person['module']} | likes {', '.join(person['interests'])}",
                 font=ctk.CTkFont(size=13), text_color="gray70",
                 wraplength=580, justify="left").pack(anchor="w", padx=16, pady=(0, 12))

    for rank, (buddy, s, shared, same_module) in enumerate(matches, 1):
        card = ctk.CTkFrame(scroll, corner_radius=12)
        card.pack(fill="x", padx=16, pady=6)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(12, 0))
        ctk.CTkLabel(top, text=f"#{rank}  {buddy['name']}",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkLabel(top, text=f"{s:.0%} match", text_color=ACCENT,
                     font=ctk.CTkFont(size=16, weight="bold")).pack(side="right")

        ctk.CTkLabel(card, text=f"{buddy['dept']} | {buddy['module']} | age {buddy['age']}",
                     text_color="gray70").pack(anchor="w", padx=14)
        bar = ctk.CTkProgressBar(card, progress_color=ACCENT)
        bar.set(s)
        bar.pack(fill="x", padx=14, pady=8)
        ctk.CTkLabel(card, text=explain(buddy, shared, same_module),
                     wraplength=560, justify="left").pack(anchor="w", padx=14, pady=(0, 12))

    ctk.CTkLabel(scroll, text="Community buddy network",
                 font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=16, pady=(18, 6))
    with Image.open(OUT_PATH) as im:
        img = im.copy()
    state["graph_img"] = ctk.CTkImage(light_image=img, dark_image=img, size=(480, 480))
    ctk.CTkLabel(scroll, image=state["graph_img"], text="").pack(pady=6)
    ctk.CTkLabel(scroll, text=f"Saved to {OUT_PATH}", text_color="gray60",
                 font=ctk.CTkFont(size=11), wraplength=560).pack(pady=(0, 16))

PAGES = [page_welcome, page_about, page_interests, page_results]


# ---------------- page switching ----------------

def show():
    for w in body.winfo_children():
        w.destroy()
    step = state["step"]
    error_label.configure(text="")
    step_label.configure(text=f"Step {step + 1} of {len(PAGES)}: {TITLES[step]}")
    progress.set(step / (len(PAGES) - 1))
    back_btn.configure(state="normal" if 0 < step < 3 else "disabled")
    next_btn.configure(text=["Start survey", "Next", "Find my buddies", "New survey"][step])
    PAGES[step]()


show()
root.mainloop()