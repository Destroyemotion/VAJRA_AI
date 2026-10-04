"""
Generates india_tier_3_city_coordinates.xlsx with curated Indian Tier-3 cities.
Follows expected sheet format: sheet_name="Coordinates", skiprows=4 header.
"""

import pandas as pd
import openpyxl

tier3_cities = [
    {"City": "Ajmer", "State": "Rajasthan", "Latitude": 26.4499, "Longitude": 74.6399},
    {"City": "Udaipur", "State": "Rajasthan", "Latitude": 24.5854, "Longitude": 73.7125},
    {"City": "Kota", "State": "Rajasthan", "Latitude": 25.2138, "Longitude": 75.8648},
    {"City": "Bikaner", "State": "Rajasthan", "Latitude": 28.0229, "Longitude": 73.3119},
    {"City": "Aligarh", "State": "Uttar Pradesh", "Latitude": 27.8974, "Longitude": 78.0880},
    {"City": "Gorakhpur", "State": "Uttar Pradesh", "Latitude": 26.7606, "Longitude": 83.3732},
    {"City": "Jhansi", "State": "Uttar Pradesh", "Latitude": 25.4484, "Longitude": 78.5685},
    {"City": "Mathura", "State": "Uttar Pradesh", "Latitude": 27.4924, "Longitude": 77.6737},
    {"City": "Gaya", "State": "Bihar", "Latitude": 24.7914, "Longitude": 85.0002},
    {"City": "Bhagalpur", "State": "Bihar", "Latitude": 25.2425, "Longitude": 86.9842},
    {"City": "Muzaffarpur", "State": "Bihar", "Latitude": 26.1209, "Longitude": 85.3647},
    {"City": "Cuttack", "State": "Odisha", "Latitude": 20.4625, "Longitude": 85.8828},
    {"City": "Rourkela", "State": "Odisha", "Latitude": 22.2604, "Longitude": 84.8536},
    {"City": "Bilaspur", "State": "Chhattisgarh", "Latitude": 22.0797, "Longitude": 82.1391},
    {"City": "Warangal", "State": "Telangana", "Latitude": 17.9689, "Longitude": 79.5941},
    {"City": "Nizamabad", "State": "Telangana", "Latitude": 18.6725, "Longitude": 78.0941},
    {"City": "Salem", "State": "Tamil Nadu", "Latitude": 11.6643, "Longitude": 78.1460},
    {"City": "Vellore", "State": "Tamil Nadu", "Latitude": 12.9165, "Longitude": 79.1325},
    {"City": "Tirunelveli", "State": "Tamil Nadu", "Latitude": 8.7139, "Longitude": 77.7567},
    {"City": "Kolhapur", "State": "Maharashtra", "Latitude": 16.7050, "Longitude": 74.2433},
    {"City": "Solapur", "State": "Maharashtra", "Latitude": 17.6599, "Longitude": 75.9064},
    {"City": "Nanded", "State": "Maharashtra", "Latitude": 19.1383, "Longitude": 77.3210},
    {"City": "Amravati", "State": "Maharashtra", "Latitude": 20.9374, "Longitude": 77.7796},
    {"City": "Ujjain", "State": "Madhya Pradesh", "Latitude": 23.1765, "Longitude": 75.7885},
    {"City": "Sagar", "State": "Madhya Pradesh", "Latitude": 23.8388, "Longitude": 78.7378},
    {"City": "Rewa", "State": "Madhya Pradesh", "Latitude": 24.5362, "Longitude": 81.3037},
    {"City": "Rohtak", "State": "Haryana", "Latitude": 28.8955, "Longitude": 76.6066},
    {"City": "Karnal", "State": "Haryana", "Latitude": 29.6857, "Longitude": 76.9905},
    {"City": "Shimla", "State": "Himachal Pradesh", "Latitude": 31.1048, "Longitude": 77.1734},
    {"City": "Durgapur", "State": "West Bengal", "Latitude": 23.5204, "Longitude": 87.3119},
    {"City": "Siliguri", "State": "West Bengal", "Latitude": 26.7271, "Longitude": 88.3953},
    {"City": "Asansol", "State": "West Bengal", "Latitude": 23.6889, "Longitude": 86.9661}
]

def build_excel_file():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Coordinates"

    # Add 4 title/metadata rows
    ws.append(["VAJRA-AI Indian Tier-3 City Coordinates Dataset"])
    ws.append(["Ministry of Earth Sciences & ISRO Atmospheric Nowcasting"])
    ws.append(["Source: Survey of India & Open-Meteo Geocoding"])
    ws.append([]) # blank line before header (skiprows=4)

    # Row 5: Headers
    ws.append(["City", "State", "Latitude", "Longitude"])

    # Row 6+: City Data
    for c in tier3_cities:
        ws.append([c["City"], c["State"], c["Latitude"], c["Longitude"]])

    wb.save("india_tier_3_city_coordinates.xlsx")
    print(f"Created india_tier_3_city_coordinates.xlsx with {len(tier3_cities)} cities.")

if __name__ == "__main__":
    build_excel_file()
