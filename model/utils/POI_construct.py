"""Utilities for constructing and maintaining the user-owned POI database.

This module implements an automated pipeline that mirrors the workflow
described in the ITINERA paper (§B/§E). The end-to-end process ingests
travel-related social media posts, enriches the raw content with OCR/ASR,
extracts POIs via GPT-3.5, resolves their locations through the Amap API,
and finally generates descriptions and embeddings before writing the
records into the POI database :math:`P` and embedding matrix :math:`E`.

The implementation focuses on XiaoHongShu (小红书) posts, leveraging the
provided scraper stub. The abstractions are intentionally modular so the
same pipeline can be extended to other sources (e.g., 马蜂窝、公众号) by
plugging different scrapers.
"""

from __future__ import annotations

from csv import reader
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

from Scraper.xhs_scraper import XHSScraper
from paddleocr import PaddleOCR

try:  # Local import to avoid circular dependency when running unit tests.
    from model.utils.proxy_call import OpenaiCall
    from model.utils.all_en_prompts import (
        get_poi_description_prompt,
        get_poi_extraction_prompt,
    )
except ImportError:  # pragma: no cover - handled during runtime.
    OpenaiCall = None  # type: ignore


LOGGER = logging.getLogger(__name__)


def _lonlat_to_webmercator(lon: float, lat: float) -> tuple[float, float]:
    """Converts GCJ-02/WGS-84 longitude & latitude to Web Mercator (EPSG:3857)."""

    radius = 6378137.0
    x = math.radians(lon) * radius
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * radius
    return x, y


@dataclass
class ScrapedPost:
    """Normalized payload returned by :class:`XHSScraper`."""

    url: str
    title: str
    content: str
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    tags: Optional[List[str]] = None
    author_city: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def merged_text(self) -> str:
        """Combines textual fields into a single block that LLMs can digest."""

        sections = [self.title.strip(), self.content.strip()]
        if self.tags:
            sections.append("Tags: " + ", ".join(self.tags))
        if self.meta.get("location"):
            sections.append(f"Author tagged location: {self.meta['location']}")
        return "\n\n".join([s for s in sections if s])


@dataclass
class POIRecord:
    """Represents a single POI entry before persisting to the database."""

    name: str
    location_text: Optional[str]
    longitude: Optional[float]
    latitude: Optional[float]
    amap_id: Optional[str]
    city: Optional[str]
    address: Optional[str]
    source_url: str
    source_context: str
    category: Optional[str] = None
    rating: Optional[str] = None
    description: Optional[str] = None

    def formatted_context(self) -> str:
        """Builds the context text stored in the CSV and embedding."""
        coord_text = (
            f"{self.latitude:.6f}, {self.longitude:.6f}"
            if self.latitude is not None and self.longitude is not None
            else "unknown"
        )
        parts = [
            self.description or self.source_context,
            "Integration of POI data via the Amap API.",
            f"Address: {self.address or self.location_text or 'unknown'}",
            f"Coordinates: {coord_text}",
        ]
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.rating:
            parts.append(f"User rating: {self.rating}")
        return " ".join(part for part in parts if part)

    def to_row(self, poi_id: int) -> Dict[str, object]:
        if self.longitude is None or self.latitude is None:
            raise ValueError(
                f"POI '{self.name}' is missing coordinates and cannot be stored."
            )

        x, y = _lonlat_to_webmercator(self.longitude, self.latitude)
        # final_context = self.formatted_context()
        return {
            "id": poi_id,
            "name": self.name,
            "x": x,
            "y": y,
            "lon": self.longitude,
            "lat": self.latitude,
            "context": self.description or self.source_context,
        }


class MediaContentAggregator:
    """Handles OCR (image) and ASR (video) enrichment for scraped posts."""

    def __init__(
        self,
        asr_model_name: str = "base",
    ):
        self.asr_model_name = asr_model_name
        self.session = requests.Session()
        self._whisper_model = None
        self._paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract_image_text(self, image_urls: Sequence[str]) -> List[str]:
        texts: List[str] = []

        if not image_urls:
            return texts

        for idx, url in enumerate(image_urls):
            try:
        
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                np_img = np.array(img)
                result = self._paddle_ocr.predict(input=np_img)
                rec_texts = result[0]["rec_texts"]
                text = " ".join(rec_texts).strip()
                if text:
                    texts.append(text)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("OCR failed for %s (image %d): %s", url, idx, exc)
        return texts

    def transcribe_videos(self, video_urls: Sequence[str]) -> List[str]:
        transcripts: List[str] = []
        for idx, url in enumerate(video_urls):
            tmp_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                    with self.session.get(url, stream=True, timeout=30) as resp:
                        resp.raise_for_status()
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                tmp.write(chunk)
                    tmp.flush()

                transcript = self._run_whisper(tmp_path)
                if transcript:
                    transcripts.append(transcript)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("ASR failed for %s (video %d): %s", url, idx, exc)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        return transcripts

    def _run_whisper(self, media_path: str) -> str:
        try:
            import whisper  # Lazy import to avoid mandatory dependency during unit tests
        except ImportError as exc:  # pragma: no cover
            LOGGER.error("whisper package is required for ASR: %s", exc)
            return ""

        if self._whisper_model is None:
            self._whisper_model = whisper.load_model(self.asr_model_name)

        result = self._whisper_model.transcribe(media_path)
        return result.get("text", "").strip()


class AmapClient:
    """Thin wrapper around the 高德地图 (Amap) text search API."""

    SEARCH_URL = "https://restapi.amap.com/v3/place/text"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Amap API key is required for POI lookup.")
        self.api_key = api_key
        self.session = requests.Session()

    def lookup(self, name: str, city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params = {
            "key": self.api_key,
            "keywords": name,
            "city": city or "",
            "children": 0,
            "offset": 1,
            "page": 1,
            "extensions": "base",
            "output": "json",
        }
        try:
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:  # pragma: no cover - network call.
            LOGGER.error("Amap lookup failed for %s: %s", name, exc)
            return None

        if payload.get("status") != "1" or not payload.get("pois"):
            return None

        entry = payload["pois"][0]
        location = entry.get("location", "")
        try:
            lon_str, lat_str = location.split(",")
            lon, lat = float(lon_str), float(lat_str)
        except ValueError:
            lon, lat = None, None  # type: ignore[assignment]

        biz_ext = entry.get("biz_ext") or {}
        return {
            "name": entry.get("name") or name,
            "address": entry.get("address"),
            "amap_id": entry.get("id"),
            "longitude": lon,
            "latitude": lat,
            "category": entry.get("type"),
            "rating": biz_ext.get("rating"),
        }


class POIDatabase:
    """Manages persistence for the POI table :math:`P` and embedding matrix :math:`E`."""

    COLUMNS = ["id", "name", "x", "y", "lon", "lat", "context"]

    def __init__(self, poi_csv_path: str | Path, embedding_path: str | Path):
        self.poi_csv_path = Path(poi_csv_path)
        self.embedding_path = Path(embedding_path)
        self.poi_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_path.parent.mkdir(parents=True, exist_ok=True)

    def load_table(self) -> pd.DataFrame:
        if self.poi_csv_path.exists():
            df = pd.read_csv(self.poi_csv_path)
        else:
            df = pd.DataFrame(columns=self.COLUMNS)

        for col in self.COLUMNS:
            if col not in df.columns:
                if col in {"name", "context"}:
                    df[col] = ""
                else:
                    df[col] = np.nan
        return df

    def load_embeddings(self) -> np.ndarray:
        if self.embedding_path.exists():
            return np.load(self.embedding_path)
        return np.empty((0, 0))

    def append(self, poi_rows: List[Dict[str, object]], embeddings: np.ndarray) -> None:
        if not poi_rows:
            LOGGER.info("No POIs to append; skipping persistence.")
            return

        df = self.load_table()
        start_id = int(df["id"].max()) + 1 if not df.empty else 0
        for idx, row in enumerate(poi_rows):
            row["id"] = start_id + idx

        updated = pd.concat([df, pd.DataFrame(poi_rows)], ignore_index=True)
        updated = updated[self.COLUMNS]
        updated.to_csv(self.poi_csv_path, index=False)

        existing_embeddings = self.load_embeddings()
        if existing_embeddings.size == 0:
            combined = embeddings
        else:
            combined = np.vstack([existing_embeddings, embeddings])
        np.save(self.embedding_path, combined)


class POIConstructionPipeline:
    """End-to-end orchestrator for generating POIs from XiaoHongShu posts."""

    def __init__(
        self,
        amap_api_key: Optional[str],
        poi_csv_path: str | Path,
        embedding_path: str | Path,
        proxy_call: Optional[OpenaiCall] = None,
        city: Optional[str] = None,
        extraction_model: str = "gpt-3.5-turbo",
        description_model: str = "gpt-3.5-turbo",
        asr_model: str = "base",
    ):
        if proxy_call is None:
            if OpenaiCall is None:
                raise RuntimeError(
                    "OpenaiCall is unavailable; cannot initialize pipeline."
                )
            proxy_call = OpenaiCall()

        self.proxy = proxy_call
        self.openai_client = getattr(self.proxy, "client", None)
        if self.openai_client is None:
            raise RuntimeError(
                "The provided proxy_call must expose an OpenAI client via `.client`."
            )
        if amap_api_key is None:
            amap_api_key = os.getenv("AMAP_API_KEY")
        if not amap_api_key:
            raise RuntimeError(
                "Amap API key is required to initialize the POIConstructionPipeline."
            )

        self.scraper = XHSScraper()
        self.media = MediaContentAggregator(
            asr_model_name=asr_model,
        )
        self.amap = AmapClient(amap_api_key)
        self.database = POIDatabase(poi_csv_path, embedding_path)
        self.default_city = city
        self.extraction_model = extraction_model
        self.description_model = description_model

    @classmethod
    def from_city(
        cls,
        city_name: str,
        amap_api_key: Optional[str] = None,
        proxy_call: Optional[OpenaiCall] = None,
        extraction_model: str = "gpt-3.5-turbo",
        description_model: str = "gpt-3.5-turbo",
        asr_model: str = "base",
    ) -> "POIConstructionPipeline":
        data_dir = Path("model") / "data"
        poi_csv_path = data_dir / f"{city_name}_zh.csv"
        embedding_path = data_dir / f"{city_name}_zh.npy"
        return cls(
            amap_api_key=amap_api_key,
            poi_csv_path=poi_csv_path,
            embedding_path=embedding_path,
            proxy_call=proxy_call,
            city=city_name,
            extraction_model=extraction_model,
            description_model=description_model,
            asr_model=asr_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_on_posts(self, urls: Iterable[str]) -> List[POIRecord]:
        """Processes multiple XiaoHongShu posts and appends POIs to the database."""

        all_records: List[POIRecord] = []
        for url in urls:
            records = self.process_single_post(url)
            all_records.extend(records)
        return all_records

    def process_single_post(self, url: str) -> List[POIRecord]:
        """Main entry point for a single post."""

        print("\n" + "=" * 80)
        print("🚀 Starting POI Construction Pipeline")
        print("=" * 80)
        print(f"📍 Processing URL: {url}")

        print("\n[Step 1/6] 🕷️  Scraping post content...")
        scraped = self._scrape(url)
        print(
            f"✓ Title: {scraped.title[:50]}..."
            if len(scraped.title) > 50
            else f"✓ Title: {scraped.title}"
        )
        print(f"✓ Found {len(scraped.images)} images, {len(scraped.videos)} videos")
        
        # Print detailed ScrapedPost information
        print("\n📋 Scraped Post Details:")
        print(f"  • URL: {scraped.url}")
        print(f"  • Title: {scraped.title}")
        print(f"  • Content: {scraped.content[:200]}..." if len(scraped.content) > 200 else f"  • Content: {scraped.content}")
        print(f"  • Tags: {scraped.tags}")
        print(f"  • Author City: {scraped.author_city}")
        print(f"  • Location (meta): {scraped.meta.get('location')}")
        print(f"  • Number of Images: {len(scraped.images)}")
        print(f"  • Number of Videos: {len(scraped.videos)}")

        print("\n[Step 2/6] 🖼️  Extracting text from images (OCR)...")
        ocr_texts = self.media.extract_image_text(scraped.images)
        print(f"✓ Extracted text from {len(ocr_texts)} images")

        # transcripts = self.media.transcribe_videos(scraped.videos)
        transcripts = []

        print("\n[Step 3/6] 📝 Merging content from all sources...")
        unified_context = self._merge_modal_content(scraped, ocr_texts, transcripts)
        print(f"✓ Unified context length: {len(unified_context)} characters")

        print("\n[Step 4/6] 🔍 Extracting POI candidates using LLM...")
        poi_candidates = self._extract_pois(unified_context)
        print(f"✓ Found {len(poi_candidates)} POI candidates")

        print("\n[Step 5/6] 🗺️  Resolving locations via Amap API...")
        resolved_pois = self._resolve_locations(poi_candidates, scraped)
        print(f"✓ Successfully resolved {len(resolved_pois)} POIs with coordinates")
        
        # Print detailed POI information from Amap
        if resolved_pois:
            print("\n📍 Resolved POI Details from Amap:")
            for idx, poi in enumerate(resolved_pois, 1):
                print(f"\n  POI #{idx}:")
                print(f"    • Name: {poi.name}")
                print(f"    • Address: {poi.address}")
                print(f"    • City: {poi.city}")
                print(f"    • Category: {poi.category}")
                print(f"    • Rating: {poi.rating}")
                print(f"    • Coordinates: ({poi.longitude:.6f}, {poi.latitude:.6f})" if poi.longitude and poi.latitude else "    • Coordinates: N/A")
                print(f"    • Amap ID: {poi.amap_id}")
                print(f"    • Location Text: {poi.location_text}")

        print("\n[Step 6/6] ✍️  Generating descriptions using LLM...")
        enriched_pois = self._generate_descriptions(resolved_pois, unified_context)
        print(f"✓ Generated descriptions for {len(enriched_pois)} POIs")

        print("\n[Final] 💾 Validating and preparing records...")
        valid_records: List[POIRecord] = []
        rows: List[Dict[str, object]] = []
        for record in enriched_pois:
            try:
                rows.append(record.to_row(poi_id=0))
                valid_records.append(record)
            except ValueError as exc:
                LOGGER.warning("Skipping POI %s: %s", record.name, exc)

        if valid_records:
            print(f"✓ Validated {len(valid_records)} POI records")

            print("\n[Embeddings] 🧮 Creating embeddings...")
            embedding_texts = []
            for record in valid_records:
                coord_text = (
                    f"{record.latitude:.6f}, {record.longitude:.6f}"
                    if record.latitude is not None and record.longitude is not None
                    else "unknown"
                )
                embedding_texts.append(
                    f"{record.name}: Integration of POI data via the Amap API. Address: {record.address or record.location_text or 'unknown'}, Coordinates: ({coord_text}), Category: {record.category or 'unspecified'}, Rating: {record.rating or 'N/A'}. Details: {record.description or unified_context}"
                )
            embeddings = self._create_embeddings(embedding_texts)
            print(f"✓ Created {len(embeddings)} embeddings")

            print("\n[Database] 💽 Saving to database...")
            self.database.append(rows, embeddings)
            print(f"✓ Successfully saved {len(valid_records)} POIs to database")

            print("\n" + "=" * 80)
            print("✅ POI Construction Pipeline completed successfully!")
            print("=" * 80 + "\n")
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  No valid POIs identified for {url}")
            print("=" * 80 + "\n")
            LOGGER.info("No POIs identified for %s", url)
        return valid_records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _scrape(self, url: str) -> ScrapedPost:
        payload = self.scraper.extract_post_content(url)

        videos = payload.get("video_urls") or []
        author_info = payload.get("author") or {}
        return ScrapedPost(
            url=payload["url"],
            title=payload.get("title", ""),
            content=payload.get("content", ""),
            images=payload.get("img_urls", []) or [],
            videos=videos,
            tags=payload.get("tags"),
            author_city=author_info.get("city"),
            meta={"location": payload.get("location")},
        )

    def _merge_modal_content(
        self, scraped: ScrapedPost, ocr_texts: Sequence[str], transcripts: Sequence[str]
    ) -> str:
        pieces = [scraped.merged_text]
        if ocr_texts:
            pieces.append("OCR extracted text:\n" + "\n".join(ocr_texts))
        if transcripts:
            pieces.append("Video transcripts:\n" + "\n".join(transcripts))
        return "\n\n".join([p for p in pieces if p.strip()])

    def _extract_pois(self, context: str) -> List[Dict[str, Optional[str]]]:
        """Extract POI names and addresses from context using LLM.
        Output format:
        { "POI Name": "Related Address Information for the POI" }
        """
        prompt = get_poi_extraction_prompt(post_info=context)
        response = self.proxy.chat(
            messages=[{"role": "user", "content": prompt}], model=self.extraction_model
        )

        # Clean up the response: remove markdown code blocks and special characters
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block markers
            lines = response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines)

        # Replace non-breaking spaces
        response = response.replace("\xa0", " ")
        
        
        # Log the cleaned response for debugging
        print(f"\n[DEBUG] Cleaned LLM response:\n{response}\n")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            LOGGER.error("Failed to parse POI extraction response: %s", response)
            return []

        pois: List[Dict[str, Optional[str]]] = []
        if isinstance(parsed, dict):
            items = parsed.items()
        elif isinstance(parsed, list):
            # Gracefully handle list of {name: address}
            items = []
            for entry in parsed:
                if isinstance(entry, dict):
                    items += entry.items()
        else:
            items = []

        for name, address in items:
            if not name:
                continue
            pois.append({"name": str(name).strip(), "address": address})
        return pois

    def _resolve_locations(
        self, raw_pois: List[Dict[str, Optional[str]]], scraped: ScrapedPost
    ) -> List[POIRecord]:
        resolved: List[POIRecord] = []
        for item in raw_pois:
            name = item.get("name")
            if not name:
                continue
            city = item.get("city") or scraped.author_city or self.default_city
            amap_result = self.amap.lookup(name, city=city)
            longitude = amap_result.get("longitude") if amap_result else None
            latitude = amap_result.get("latitude") if amap_result else None
            amap_id = amap_result.get("amap_id") if amap_result else None
            address = amap_result.get("address") if amap_result else item.get("address")
            category = amap_result.get("category") if amap_result else None
            rating = amap_result.get("rating") if amap_result else None
            # if lon or lat is None, skip this POI
            if longitude is None or latitude is None:
                continue
            resolved.append(
                POIRecord(
                    name=name,
                    location_text=item.get("address"),
                    longitude=longitude,
                    latitude=latitude,
                    amap_id=amap_id,
                    city=city,
                    address=address,
                    category=category,
                    rating=rating,
                    description=None,
                    source_url=scraped.url,
                    source_context=scraped.merged_text,
                )
            )
        return resolved

    def _generate_descriptions(
        self, poi_records: List[POIRecord], context: str
    ) -> List[POIRecord]:
        if not poi_records:
            return []

        # Prepare POI data with name, address, and category
        poi_data = []
        for record in poi_records:
            poi_data.append(
                {
                    "name": record.name,
                    "address": record.address or "",
                    "category": record.category or "",
                }
            )

        prompt = get_poi_description_prompt(post_info=context, poi_data=poi_data)
        response = self.proxy.chat(
            messages=[{"role": "user", "content": prompt}], model=self.description_model
        )

        # Clean up the response: remove markdown code blocks and special characters
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines)
        response = response.replace("\xa0", " ")

        try:
            desc_map = json.loads(response)
        except json.JSONDecodeError:
            LOGGER.error("Failed to parse POI description response: %s", response)
            desc_map = {}

        for record in poi_records:
            desc = desc_map.get(record.name) if isinstance(desc_map, dict) else None
            if isinstance(desc, str) and desc.lower() != "null" and desc.strip():
                record.description = desc.strip()
            else:
                record.description = ""
        return poi_records

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        res_records = self.proxy.embedding(input_data=texts)
        embeddings = [np.array(record.embedding) for record in res_records.data]
        return np.array(embeddings, dtype=np.float32)


__all__ = [
    "POIConstructionPipeline",
    "POIDatabase",
    "POIRecord",
    "ScrapedPost",
    "MediaContentAggregator",
    "AmapClient",
]


def _demo_pipeline_run() -> None:
    """Runs the full pipeline on a sample XiaoHongShu post for manual testing."""


    #demo_url = ("https://www.xiaohongshu.com/explore/67dd8119000000001d02f76d?xsec_token=ABSO6S-bEcyA44-o3ESfto_kYILTRAz6QseFGnMcfzWY0=&xsec_source=pc_search&source=unknown")
    demo_url = ("https://www.xiaohongshu.com/explore/67ee5d1d000000001e00246e?xsec_token=ABmFOVPjsGOWKxvhyFPG-eudTimkU-pLb5QMC9xQhaGhY=&xsec_source=pc_search&source=unknown")
    city_name = os.getenv("ITINERA_CITY", "shanghai")
    amap_key = os.getenv("AMAP_API_KEY")

    if not amap_key:
        raise RuntimeError("AMAP_API_KEY is required to run the demo pipeline.")
    if OpenaiCall is None:
        raise RuntimeError("OpenAI dependency is missing; cannot run demo pipeline.")

    pipeline = POIConstructionPipeline.from_city(
        city_name=city_name,
        amap_api_key=amap_key,
        proxy_call=OpenaiCall(),
    )
    records = pipeline.run_on_posts([demo_url])
    LOGGER.info("Demo pipeline inserted %d POIs for %s", len(records), city_name)


if __name__ == "__main__":
    _demo_pipeline_run()
