import requests
import dotenv
import os
import json
from turf import distance, point, centroid, polygon, boolean_point_in_polygon, bbox
from geographiclib.geodesic import Geodesic
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.geocoders import TomTom
import re
import time

dotenv.load_dotenv()

geocoder = TomTom(os.environ['TOMTOM_API_KEY'])

geo = Geodesic.WGS84

constants = json.load(open("constants.json", "r"))

polygon_points = json.load(open("polygon.json", "r"))
POLYGON_RANGE = polygon([polygon_points + [polygon_points[0]]])

TARGET_POINTS = [point(x[:2], properties={'name': x[2]}) for x in json.load(open("targets.json", "r"))]
TARGET_POINTS_COORDS = [p['geometry']['coordinates'] for p in TARGET_POINTS]

ONTARIO_HIGHWAY_REGEX = re.compile(r"ON-\d+.*")

def find_existing_chargers(quantity: int = None, options: dict = {}, center_lat: float = constants['DEFAULT_CENTER_LAT'], center_long: float = constants['DEFAULT_CENTER_LONG'], radius: float = constants['DEFAULT_SEARCH_RADIUS'], format_: str = 'json'):
    params = {
        'country': 'CA',
        'owner_type': 'all',
        'cards_accepted': 'all',
        'fuel_type': 'ELEC',
        'access': 'public',
        'status': 'E,T',
        'ev_charging_level': 'all',
        'ev_connector_type': 'all',
        'ev_network': 'all',
        'latitude': center_lat,
        'longitude': center_long,
        'radius': radius,
        'limit': 'all' if quantity == None else quantity,
    } | options
    params['api_key'] = os.environ['NREL_API_KEY']
    r = requests.get(f"https://developer.nrel.gov/api/alt-fuel-stations/v1/nearest.{format_}", params=params)
    return json.loads(r.content)

def find_approximate_needed_charger_locations_centroid(quantity: int = None, min_distance: float = 3):
    d = find_existing_chargers()
    existing_points_list = [point([v['longitude'], v['latitude']]) for v in d['fuel_stations']]
    centroid_points_set = set(sum([[tuple(centroid([p, k])['geometry']['coordinates']) for k in existing_points_list if p != k] for p in existing_points_list], []))
    centroid_points_list = [[p[0], p[1], min([distance(point(list(p)), k, options={'units': 'miles'}) for k in existing_points_list if p != k])] for p in centroid_points_set]
    centroid_points_list = sorted(centroid_points_list, key=lambda p: p[2], reverse=True)
    return [[x[1], x[0], x[2]] for x in centroid_points_list if x[2] >= min_distance][:quantity]

def find_approximate_needed_charger_locations_targets(distance_from_targets: float = 1):
    ret = []
    radius_meters = distance_from_targets * 1609.34
    for target in TARGET_POINTS_COORDS:
        coords = lambda: target
        lat_bounds = sorted([geo.Direct(*coords(), -180, radius_meters)['lat2'], geo.Direct(*coords(), 0, radius_meters)['lat2']])
        long_bounds = sorted([geo.Direct(*coords(), 90, radius_meters)['lon2'], geo.Direct(*coords(), -90, radius_meters)['lon2']])

        [ret.append(x + ("artificial",)) for x in [(long_bounds[0], lat_bounds[0]), (long_bounds[0], lat_bounds[1]), (long_bounds[1], lat_bounds[0]), (long_bounds[1], lat_bounds[1]), tuple(target)] if boolean_point_in_polygon(point(list(x)), POLYGON_RANGE)]
    return ret

def find_approximate_needed_charger_locations_newcentroid(quantity: int = None, distance_from_targets: float = 1, max_tolerable_existing_stations: int = 3, hard_min_distance: float = 1, min_distance: float = 3, max_sub_distance: float = 1, min_sub_count: int = 3):
    d = find_existing_chargers(format_='geojson')

    centroid_points_list = set(sum([[tuple(cent['geometry']['coordinates']) for f2 in d['features'] if boolean_point_in_polygon((cent := centroid([f1, f2])), POLYGON_RANGE) and f2 != f1] for f1 in d['features']], []))

    #ret_points = [expr for v in ret_points if (expr := v + (min([distance(point(list(v)), f, options={'units': 'miles'}) for f in d['features']]),))[2] >= min_distance]

    #ret_points = [v + (min(expr), len(expr)) for v in ret_points if len(list(filter(lambda r: r < min_distance, (expr := [dist for f in (d['features'] + ret_points) if v != f and (dist := distance(point(list(v)), f if isinstance(f, dict) else point(list(f)), options={'units': 'miles'}))])))) <= max_tolerable_existing_stations]
    
    union_points = centroid_points_list
    ret_points = []
    excl_points = set()
    for p in union_points:
        if p in excl_points:
            continue
        close_points = [(dist, f) for f in list(union_points - excl_points) + d['features'] if (dist := distance(point(list(p[:2])), f if isinstance(f, dict) else point(list(f[:2])))) < min_distance]
        if len(close_points) > max_tolerable_existing_stations:
            #excl_points |= {p}
            excl_points |= set([c[1] for c in close_points if (not isinstance(c[1], dict)) and (len(c[1]) < 3 or c[1][2] != 'artificial')])
        else:
            ret_points.append(p)
            excl_points |= set([c[1] for c in close_points if (not isinstance(c[1], dict)) and (len(c[1]) < 3 or c[1][2] != 'artificial') and c[0] < hard_min_distance])

    ret_points = list(ret_points) + find_approximate_needed_charger_locations_targets(distance_from_targets=distance_from_targets)

    remapped_points = []
    for p in ret_points:
        loc = geocoder.reverse(f"{p[1]},{p[0]}")
        if re.match(ONTARIO_HIGHWAY_REGEX, loc.address):
            continue
        remapped_points.append(point([loc.longitude, loc.latitude], properties={'address': loc.address}))
        time.sleep(0.5) # Prevent rate limit

    for p in remapped_points:
        p['properties']['distance_to_existing'] = min([distance(p, f) for f in d['features']])

    remapped_points = sorted(remapped_points, key=lambda k: k['properties']['distance_to_existing'], reverse=True)

    return remapped_points[:quantity]

def find_approximate_needed_charger_locations_testpoints(quantity: int = None, min_distance: float = 3, max_sub_distance: float = 3, min_sub_distance_count: int = 15, test_point_step: float = 0.02, radius: float = constants['DEFAULT_SEARCH_RADIUS'], center_lat: float = constants['DEFAULT_CENTER_LAT'], center_long: float = constants['DEFAULT_CENTER_LONG']):
    d = find_existing_chargers(format_='geojson')
    #radius_meters = radius * 1609.34
    #lat_bounds = sorted([geo.Direct(center_lat, center_long, -180, radius_meters)['lat2'], geo.Direct(center_lat, center_long, 0, radius_meters)['lat2']])
    #long_bounds = sorted([geo.Direct(center_lat, center_long, 90, radius_meters)['lon2'], geo.Direct(center_lat, center_long, -90, radius_meters)['lon2']])

    #lat_range = np.arange(*lat_bounds, step=test_point_step)
    #long_range = np.arange(*long_bounds, step=test_point_step)

    bounds = bbox(POLYGON_RANGE) # More efficient
    lat_range = np.arange(bounds[1], bounds[3], step=test_point_step)
    long_range = np.arange(bounds[0], bounds[2], step=test_point_step)

    # (geo.Inverse(y, x, *reversed(v['geometry']['coordinates']))['s12'] / 1609.34) # Alternative distance calc for below
    test_point_list = sum([[(y, x, min([distance([x, y], v['geometry']['coordinates'], options={'units': 'miles'}) for v in d['features']])) for y in lat_range] for x in long_range], [])
    
    # Can be used to generate heat map when sorted this way
    test_point_list = sorted([x for x in test_point_list if x[2] >= min_distance and boolean_point_in_polygon([x[1], x[0]], POLYGON_RANGE)], key = lambda q: q[2], reverse=True)

    # Clustering
    #yhat = DBSCAN(eps=0.30, min_samples=9).fit_predict(test_point_list)
    #test_point_list = [x + (yhat[ind],) for ind, x in enumerate(test_point_list)]
    
    test_point_list = sorted([expr for v in test_point_list if (expr := (v + (len([k for k in test_point_list if distance([v[1], v[0]], [k[1], k[0]], options={'units': 'miles'}) <= max_sub_distance and k != v]),)))[3] >= min_sub_distance_count], key=lambda z: z[3])
    return test_point_list

if __name__ == '__main__':
    from rich import print
