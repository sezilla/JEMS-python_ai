import json
import random
from datetime import datetime, timedelta

# define necessary dictionaries
department_teams = {
    1: [1, 2, 3, 4, 5, 6, 37],
    2: [7, 8, 9, 10, 11, 12, 38],
    3: [13, 14, 15, 16, 17, 18, 39],
    4: [19, 20, 21, 22, 23, 24, 40],
    5: [25, 26, 27, 28, 29, 30, 41],
    6: [31, 32, 33, 34, 35, 36, 42],
}

package_departments = {
    1: [1, 2, 3, 4, 5],
    2: [1, 2, 3, 4, 5, 6],
    3: [1, 2, 3, 4, 5, 6],
    4: [1, 2, 3, 4, 5, 6],
    5: [1, 2, 3, 4, 5, 6],
}

# Function to generate random project names
def random_project_name():
    first_names = ["Rigel", "Vega", "Sirius", "Altair", "Polaris", "Luna", "Nova", "Orion", "Aurora", "Lyra", "Capella", "Bellatrix", "Aldebaran", "Betelgeuse", "Deneb", "Antares", "Arcturus", "Regulus", "Spica", "Procyon", "Castor", "Pollux", "Albireo", "Mira", "Alcor", "Mizar", "Rasalhague", "Alphard", "Alphecca", "Alcyone", "Alcor", "Aludra", "Alula", "Alula Australis", "Alula Borealis", "Alya", "Alzirr", "Ancha", "Ankaa", "Anser", "Atria", "Avior", "Azha", "Baham", "Baten Kaitos", "Beemim", "Becrux", "Botein", "Canopus", "Caph", "Dabih", "Denebola", "Diadem", "Diphda", "Dubhe", "Elnath", "Enif", "Fomalhaut", "Gacrux", "Gienah", "Gomeisa", "Hadar", "Hamal", "Izar", "Kaus Australis", "Kochab", "Markab", "Menkalinan", "Menkar", "Menkent", "Merak", "Mimosa", "Mintaka", "Mira", "Mirach", "Mirfak", "Mirzam", "Mizar", "Nashira", "Nihal", "Nunki", "Peacock", "Phact", "Phecda", "Polaris", "Porrima", "Procyon", "Rasalhague", "Rastaban", "Regulus", "Rigel", "Rotanev", "Sabik", "Sadr", "Saiph", "Sargas", "Scheat", "Schedar", "Shaula", "Sheratan", "Sirius", "Spica", "Subra", "Suhail", "Sulafat", "Syrma", "Talitha", "Tania Australis", "Tania Borealis", "Tarazed", "Thuban", "Unukalhai", "Vega", "Vindemiatrix", "Wasat", "Wezen"]
    last_names = ["Event", "Wedding", "Campaign", "Gala", "Summit", "Showcase", "Expo", "Festival", "Conference", "Seminar", "Workshop", "Retreat", "Meetup", "Symposium", "Concert", "Performance", "Recital", "Gathering", "Rally", "Celebration", "Party", "Reception", "Banquet", "Dinner", "Luncheon", "Brunch", "Breakfast", "Picnic", "Barbecue", "Buffet", "Feast", "Soiree", "Shindig", "Hootenanny", "Hoedown", "Jamboree", "Fiesta", "Fete", "Carnival", "Fair", "Circus", "Parade", "Procession", "Pageant", "Spectacle", "Show", "Exhibition", "Display", "Demo", "Presentation", "Pitch", "Pitchfest", "Competition", "Contest", "Tournament", "Match", "Game"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Generate dataset
num_samples = 200
dataset = []
for _ in range(num_samples):
    package_id = random.randint(1, 5)
    start_date = datetime(2025, 2, 22) + timedelta(days=random.randint(0, 365))
    end_date = start_date + timedelta(days=random.randint(120, 548))  # 4 months (120 days) to 1.5 years (548 days)

    # Get the departments for the selected package
    departments = package_departments[package_id]

    # Allocate teams only from the valid departments
    allocated_teams = [random.choice(department_teams[d]) for d in departments]

    project = {
        "project_name": random_project_name(),
        "package_id": package_id,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "allocated_teams": allocated_teams
    }
    dataset.append(project)

# Save updated dataset to JSON
dataset_path = "synthetic_project_dataset.json"
with open(dataset_path, "w") as f:
    json.dump(dataset, f, indent=4)

dataset_path

