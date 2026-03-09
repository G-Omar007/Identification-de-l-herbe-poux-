# inaturalist.py
# Client API iNaturalist — enrichissement des données si herbe à poux détectée
# Basé sur votre Phase 1 du notebook (même API, mêmes endpoints)

import requests

BASE_URL          = "https://api.inaturalist.org/v1"
AMBROSIA_TAXON_ID = 75712   # ID iNaturalist de Ambrosia artemisiifolia
TIMEOUT_S         = 5       # Ne pas bloquer Flask si iNaturalist est lent


def get_species_info() -> dict:
    """
    Récupère les infos sur l'herbe à poux depuis iNaturalist.
    Appelé seulement quand le modèle détecte une herbe à poux (confiance > seuil).
    """
    try:
        r = requests.get(
            f"{BASE_URL}/taxa/{AMBROSIA_TAXON_ID}",
            params={"locale": "fr"},
            timeout=TIMEOUT_S
        )
        r.raise_for_status()
        d = r.json()["results"][0]

        return {
            "nom_commun":       d.get("preferred_common_name", "Herbe à poux"),
            "nom_scientifique": d["name"],
            "description":      (d.get("wikipedia_summary") or "")[:400],
            "nb_observations":  d.get("observations_count", 0),
            "photo_url":        d["default_photo"]["medium_url"]
                                if d.get("default_photo") else None,
            "wikipedia_url":    d.get("wikipedia_url"),
        }
    except Exception as e:
        print(f"[iNaturalist] get_species_info erreur: {e}")
        return {}


def get_nearby_observations(lat: float, lng: float, rayon_km: int = 10) -> list:
    """
    Trouve les observations validées d'herbe à poux proches de la position GPS.
    Même logique que votre fetch_inat_obs() du notebook (quality_grade=research).
    """
    try:
        r = requests.get(
            f"{BASE_URL}/observations",
            params={
                "taxon_id":     AMBROSIA_TAXON_ID,
                "lat":          lat,
                "lng":          lng,
                "radius":       rayon_km,
                "quality_grade":"research",
                "per_page":     10,
                "order_by":     "observed_on",
                "order":        "desc",
            },
            timeout=TIMEOUT_S
        )
        r.raise_for_status()

        obs_list = []
        for obs in r.json().get("results", []):
            coords = obs.get("geojson", {}).get("coordinates", [None, None])
            obs_list.append({
                "id":        obs.get("id"),
                "date":      obs.get("observed_on"),
                "lieu":      obs.get("place_guess"),
                "latitude":  coords[1] if coords[0] else None,
                "longitude": coords[0] if coords[0] else None,
                "url":       obs.get("uri"),
            })
        return obs_list

    except Exception as e:
        print(f"[iNaturalist] get_nearby_observations erreur: {e}")
        return []
