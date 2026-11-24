"""End-to-end and unit tests for FriendFinder backend (ML pipeline, smart discard, PNG cutout, reverse search, ZIP export)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run tests against an isolated temporary data directory and SQLite DB."""
    import app.config as cfg
    import app.database as db_mod
    import app.main as main_mod
    import app.routers.detections as r_det
    import app.routers.jobs as r_jobs
    import app.routers.media as r_media
    import app.routers.persons as r_persons
    import app.services.exporter as exporter_mod
    import app.services.extractor as extractor_mod
    import app.services.pipeline as pipeline_mod
    import app.services.search as search_mod

    data_dir = tmp_path / "data"
    dirs = {
        "DATA_DIR": data_dir,
        "UPLOADS_DIR": data_dir / "uploads",
        "FRAMES_DIR": data_dir / "frames",
        "PORTRAITS_DIR": data_dir / "portraits",
        "CUTOUTS_DIR": data_dir / "cutouts",
        "CROPS_DIR": data_dir / "crops",
        "THUMBS_DIR": data_dir / "thumbs",
        "EXPORTS_DIR": data_dir / "exports",
        # Reuse cached models under project data/models so tests don't re-download every run
        "MODELS_DIR": cfg.MODELS_DIR,
        "DB_PATH": data_dir / "friendfinder_test.db",
    }
    for d in dirs.values():
        if d.suffix != ".db":
            d.mkdir(parents=True, exist_ok=True)

    for name, val in dirs.items():
        monkeypatch.setattr(cfg, name, val)

    monkeypatch.setattr(cfg, "ALL_DATA_DIRS", [v for k, v in dirs.items() if k != "DB_PATH"])
    monkeypatch.setattr(extractor_mod, "FRAMES_DIR", dirs["FRAMES_DIR"])
    monkeypatch.setattr(extractor_mod, "PORTRAITS_DIR", dirs["PORTRAITS_DIR"])
    monkeypatch.setattr(extractor_mod, "CUTOUTS_DIR", dirs["CUTOUTS_DIR"])
    monkeypatch.setattr(extractor_mod, "CROPS_DIR", dirs["CROPS_DIR"])
    monkeypatch.setattr(extractor_mod, "THUMBS_DIR", dirs["THUMBS_DIR"])
    monkeypatch.setattr(pipeline_mod, "FRAMES_DIR", dirs["FRAMES_DIR"])
    monkeypatch.setattr(pipeline_mod, "THUMBS_DIR", dirs["THUMBS_DIR"])
    monkeypatch.setattr(r_media, "UPLOADS_DIR", dirs["UPLOADS_DIR"])
    monkeypatch.setattr(r_persons, "FRAMES_DIR", dirs["FRAMES_DIR"])
    monkeypatch.setattr(r_persons, "PORTRAITS_DIR", dirs["PORTRAITS_DIR"])
    monkeypatch.setattr(r_det, "FRAMES_DIR", dirs["FRAMES_DIR"])
    monkeypatch.setattr(r_det, "PORTRAITS_DIR", dirs["PORTRAITS_DIR"])
    monkeypatch.setattr(r_det, "THUMBS_DIR", dirs["THUMBS_DIR"])
    monkeypatch.setattr(exporter_mod, "CROPS_DIR", dirs["CROPS_DIR"])
    monkeypatch.setattr(exporter_mod, "CUTOUTS_DIR", dirs["CUTOUTS_DIR"])
    monkeypatch.setattr(exporter_mod, "EXPORTS_DIR", dirs["EXPORTS_DIR"])
    monkeypatch.setattr(exporter_mod, "FRAMES_DIR", dirs["FRAMES_DIR"])
    monkeypatch.setattr(exporter_mod, "PORTRAITS_DIR", dirs["PORTRAITS_DIR"])

    test_db = db_mod.Database(db_path=dirs["DB_PATH"])
    for mod in (db_mod, r_media, r_persons, r_det, r_jobs):
        monkeypatch.setattr(mod, "db", test_db)
    monkeypatch.setattr(pipeline_mod, "default_db", test_db)
    monkeypatch.setattr(search_mod, "default_db", test_db)
    monkeypatch.setattr(exporter_mod, "default_db", test_db)

    return {
        "db": test_db,
        "dirs": dirs,
        "client": TestClient(main_mod.app),
    }


def _find_sample_image(filename: str) -> Path:
    import insightface
    import ultralytics

    candidates = [
        Path(ultralytics.__file__).resolve().parent / "assets" / filename,
        Path(insightface.__file__).resolve().parent / "data" / "images" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Sample image not found: {filename}")


def test_quality_and_smart_discard_unit(isolated_env):
    """Verify quality scorer penalizes blur and smart discard rejects worse duplicates."""
    from app.services.ml_models import DetectedBody, DetectedFace
    from app.services.pipeline import evaluate_keep_or_discard
    from app.services.quality import score_detection

    db = isolated_env["db"]
    # Sharp textured synthetic image vs blurred version
    rng = np.random.default_rng(42)
    sharp_img = rng.integers(40, 220, size=(600, 600, 3), dtype=np.uint8)
    blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), 9.0)

    face = DetectedFace(
        bbox=[220.0, 80.0, 380.0, 260.0],
        det_score=0.95,
        embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
        pose=(2.0, 3.0, 1.0),
    )
    mask = np.zeros((600, 600), dtype=np.uint8)
    mask[70:580, 180:420] = 255
    body = DetectedBody(bbox=[180.0, 70.0, 420.0, 580.0], conf=0.92, mask=mask)

    m_sharp = score_detection(sharp_img, face, [body])
    m_blur = score_detection(blurry_img, face, [body])

    assert m_sharp.sharpness_score > m_blur.sharpness_score + 20.0
    assert m_sharp.portrait_score > m_blur.portrait_score
    assert m_sharp.matched_body is body

    # Create a person with 1 high-quality kept portrait and cap max_portraits=1
    person = db.create_person("Carlos", centroid=face.embedding)
    media = db.create_media(
        media_id="m1",
        original_filename="sharp.jpg",
        media_type="image",
        file_path="/tmp/sharp.jpg",
        file_size=100,
    )
    db.insert_detection(
        {
            "media_id": media["id"],
            "person_id": person["id"],
            "portrait_score": 88.0,
            "body_score": 85.0,
            "overall_score": 88.0,
            "has_cutout": True,
            "status": "kept",
            "embedding": face.embedding,
        }
    )

    settings = {
        "min_quality_score": "32.0",
        "max_portraits_per_person": "1",
        "max_fullbody_per_person": "1",
        "min_video_gap_seconds": "1.5",
    }
    status, _cat, reason = evaluate_keep_or_discard(
        db,
        person_id=person["id"],
        person_name="Carlos",
        media_id="m2",
        timestamp_sec=0.0,
        metrics=m_blur,
        settings=settings,
    )
    assert status == "discarded"
    assert reason and ("Calidad insuficiente" in reason or "ya tiene capturas mejores" in reason)


def test_full_ml_end_to_end_photo_video_search_export(isolated_env, tmp_path: Path):
    """Full end-to-end test using real InsightFace + YOLOv8-seg on real photos and a synthetic MP4 video."""
    from app.services.pipeline import process_media_batch_sync

    db = isolated_env["db"]
    dirs = isolated_env["dirs"]
    client = isolated_env["client"]

    # Limit top-K to 1 portrait & 1 body so we reliably trigger smart discarding on degraded duplicates
    db.update_settings(
        {
            "max_portraits_per_person": 1,
            "max_fullbody_per_person": 1,
            "video_sample_fps": 1.0,
        }
    )

    zidane_path = _find_sample_image("zidane.jpg")
    bus_path = _find_sample_image("bus.jpg")

    # 1. Create a degraded/blurred copy of zidane.jpg to verify smart discard
    zidane_bgr = cv2.imread(str(zidane_path))
    assert zidane_bgr is not None
    zidane_blur = cv2.GaussianBlur(zidane_bgr, (25, 25), 8.0)
    blur_path = tmp_path / "zidane_blurry.jpg"
    cv2.imwrite(str(blur_path), zidane_blur)

    # 2. Build a short 2-second MP4 video containing zidane.jpg (t=0s) and bus.jpg (t=1s)
    h, w = zidane_bgr.shape[:2]
    bus_bgr = cv2.imread(str(bus_path))
    assert bus_bgr is not None
    bus_resized = cv2.resize(bus_bgr, (w, h))
    video_path = tmp_path / "friends_clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (w, h),
    )
    for _ in range(10):  # 0.0s .. 0.9s -> zidane
        writer.write(zidane_bgr)
    for _ in range(10):  # 1.0s .. 1.9s -> bus
        writer.write(bus_resized)
    writer.release()
    assert video_path.exists() and video_path.stat().st_size > 1000

    # Register files and run ingestion synchronously (same pipeline called by background worker)
    m1_dest = dirs["UPLOADS_DIR"] / "m1.jpg"
    m1_dest.write_bytes(zidane_path.read_bytes())
    db.create_media(
        media_id="m1",
        original_filename="zidane.jpg",
        media_type="image",
        file_path=str(m1_dest),
        file_size=m1_dest.stat().st_size,
    )

    m2_dest = dirs["UPLOADS_DIR"] / "m2.jpg"
    m2_dest.write_bytes(blur_path.read_bytes())
    db.create_media(
        media_id="m2",
        original_filename="zidane_blurry.jpg",
        media_type="image",
        file_path=str(m2_dest),
        file_size=m2_dest.stat().st_size,
    )

    m3_dest = dirs["UPLOADS_DIR"] / "m3.mp4"
    m3_dest.write_bytes(video_path.read_bytes())
    db.create_media(
        media_id="m3",
        original_filename="friends_clip.mp4",
        media_type="video",
        file_path=str(m3_dest),
        file_size=m3_dest.stat().st_size,
    )

    job_id = db.create_job("ingest", total_items=3)
    summary = process_media_batch_sync(["m1", "m2", "m3"], job_id, database=db)

    assert summary["total_detections"] >= 3
    assert summary["kept_detections"] >= 2
    assert summary["discarded_detections"] >= 1

    # Verify blurred duplicate photo was discarded (or had 0 kept improvements)
    m2_rec = db.get_media("m2")
    assert m2_rec is not None
    assert m2_rec["kept_count"] == 0
    assert m2_rec["verdict"] in ("discarded", "no_people")

    # Verify persons list API
    resp = client.get("/api/persons")
    assert resp.status_code == 200
    persons = resp.json()
    assert len(persons) >= 2

    # Check that at least one kept detection produced all 3 formats including transparent PNG cutout
    first_person = persons[0]
    detail_resp = client.get(f"/api/persons/{first_person['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert len(detail["kept_detections"]) >= 1
    det = detail["kept_detections"][0]
    assert det["portrait_path"] and (dirs["PORTRAITS_DIR"] / det["portrait_path"]).exists()
    assert det["frame_path"] and (dirs["FRAMES_DIR"] / det["frame_path"]).exists()
    assert det["crop_path"] and (dirs["CROPS_DIR"] / det["crop_path"]).exists()
    assert det["scene_summary"]  # Instant local heuristic description populated

    cutout_dets = [
        d
        for p in persons
        for d in client.get(f"/api/persons/{p['id']}").json()["kept_detections"]
        if d.get("cutout_path")
    ]
    assert len(cutout_dets) >= 1
    cutout_file = dirs["CUTOUTS_DIR"] / cutout_dets[0]["cutout_path"]
    assert cutout_file.exists()
    cutout_rgba = cv2.imread(str(cutout_file), cv2.IMREAD_UNCHANGED)
    assert cutout_rgba is not None and cutout_rgba.shape[2] == 4  # Has transparency alpha channel!

    # Test renaming a person
    rename_resp = client.patch(
        f"/api/persons/{first_person['id']}",
        json={"name": "Zinedine"},
    )
    assert rename_resp.status_code == 200
    assert rename_resp.json()["name"] == "Zinedine"

    # Test Reverse Search by Photo -> should find matching person AND exact timestamps in friends_clip.mp4!
    with zidane_path.open("rb") as fh:
        search_resp = client.post(
            "/api/search/by-image",
            files={"file": ("query.jpg", fh, "image/jpeg")},
        )
    assert search_resp.status_code == 200
    sdata = search_resp.json()
    assert sdata["found_faces"] >= 1
    assert len(sdata["matched_persons"]) >= 1
    video_matches = [m for m in sdata["matched_media"] if m["media_id"] == "m3"]
    assert len(video_matches) == 1
    assert video_matches[0]["appearances_count"] >= 1

    # Test Game Asset ZIP Export
    export_resp = client.post(
        "/api/persons/export",
        json={
            "person_ids": [first_person["id"]],
            "include_portraits": True,
            "include_cutouts": True,
            "include_crops": True,
            "include_originals": True,
        },
    )
    assert export_resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(export_resp.content))
    names = zf.namelist()
    assert "manifest.json" in names
    assert any(n.endswith("metadata.json") for n in names)
    assert any("/portraits/" in n for n in names)

    # Test Safe Video Deletion (keeps person assets!)
    del_resp = client.delete("/api/media/m3?keep_person_assets=true")
    assert del_resp.status_code == 200
    assert not m3_dest.exists()
    assert (dirs["PORTRAITS_DIR"] / det["portrait_path"]).exists()
    assert db.get_media("m3")["source_deleted"] == 1


def test_one_hour_source_retention_and_pin_guard(isolated_env, monkeypatch: pytest.MonkeyPatch):
    """Verify default 1h expiration deletes ONLY uploaded source files (preserving generated assets),
    Mantener prevents deletion, and optional Cloudflare PIN protects API/media routes.
    """
    from datetime import datetime, timedelta, timezone
    import app.config as cfg

    db = isolated_env["db"]
    dirs = isolated_env["dirs"]
    client = isolated_env["client"]

    # 1. Create two finished uploaded source files + a generated portrait asset
    ephemeral_src = dirs["UPLOADS_DIR"] / "temp_video.mp4"
    ephemeral_src.write_bytes(b"fake-video-bytes-12345")
    kept_src = dirs["UPLOADS_DIR"] / "saved_video.mp4"
    kept_src.write_bytes(b"fake-video-bytes-keep")

    generated_portrait = dirs["PORTRAITS_DIR"] / "det1_portrait.jpg"
    generated_portrait.write_bytes(b"generated-portrait-bytes")

    m_temp = db.create_media(
        media_id="temp1",
        original_filename="temp_video.mp4",
        media_type="video",
        file_path=str(ephemeral_src),
        file_size=ephemeral_src.stat().st_size,
    )
    assert m_temp["keep_forever"] == 0
    assert m_temp["expires_at"] is not None
    db.update_media("temp1", status="done", verdict="useful")

    m_keep = db.create_media(
        media_id="keep1",
        original_filename="saved_video.mp4",
        media_type="video",
        file_path=str(kept_src),
        file_size=kept_src.stat().st_size,
    )
    db.update_media("keep1", status="done", verdict="useful")

    person = db.create_person("Marta")
    db.insert_detection(
        {
            "id": "det1",
            "media_id": "temp1",
            "person_id": person["id"],
            "portrait_score": 91.0,
            "overall_score": 91.0,
            "portrait_path": generated_portrait.name,
            "status": "kept",
        }
    )

    # User clicks "Mantener" on keep1 -> clears expires_at
    ret_resp = client.patch("/api/media/keep1/retention", json={"keep_forever": True})
    assert ret_resp.status_code == 200
    assert ret_resp.json()["keep_forever"] == 1
    assert ret_resp.json()["expires_at"] is None

    # Simulate 65 minutes passing -> purge_expired_sources deletes temp_video.mp4 ONLY
    future_iso = (datetime.now(timezone.utc) + timedelta(minutes=65)).isoformat()
    purged = db.purge_expired_sources(now_iso=future_iso)
    assert purged["purged_count"] == 1
    assert purged["purged_ids"] == ["temp1"]
    assert not ephemeral_src.exists()
    assert kept_src.exists()  # Maintained file untouched!
    assert generated_portrait.exists()  # Generated portrait preserved!
    assert db.get_media("temp1")["source_deleted"] == 1
    assert db.get_media("keep1")["source_deleted"] == 0

    # 2. Verify optional Cloudflare PIN guard
    monkeypatch.setattr(cfg, "ACCESS_PIN", "4821")
    assert client.get("/api/auth/status").json() == {"pin_required": True}
    assert client.get("/api/persons").status_code == 401
    assert client.post("/api/auth/verify", json={"pin": "0000"}).status_code == 401
    assert client.post("/api/auth/verify", json={"pin": "4821"}).status_code == 200
    assert client.get("/api/persons", headers={"X-Access-Pin": "4821"}).status_code == 200


def test_recluster_splits_mixed_identities_preserving_names(isolated_env):
    """Verify complete-linkage recluster separates distinct people mixed into one cluster while preserving custom name."""
    db = isolated_env["db"]
    client = isolated_env["client"]

    emb_katy_1 = np.zeros(512, dtype=np.float32)
    emb_katy_1[0] = 1.0
    emb_katy_2 = np.zeros(512, dtype=np.float32)
    emb_katy_2[0] = 0.96
    emb_katy_2[1] = 0.28

    emb_other_1 = np.zeros(512, dtype=np.float32)
    emb_other_1[20] = 1.0
    emb_other_2 = np.zeros(512, dtype=np.float32)
    emb_other_2[20] = 0.95
    emb_other_2[21] = 0.31

    media = db.create_media(
        media_id="mix1",
        original_filename="party.mp4",
        media_type="video",
        file_path="/tmp/party.mp4",
        file_size=100,
    )
    katy = db.create_person("Katy", centroid=emb_katy_1)

    d1 = db.insert_detection(
        {
            "media_id": media["id"],
            "person_id": katy["id"],
            "portrait_score": 92.0,
            "frontality_score": 88.0,
            "overall_score": 92.0,
            "embedding": emb_katy_1,
            "status": "kept",
        }
    )
    db.insert_detection(
        {
            "media_id": media["id"],
            "person_id": katy["id"],
            "portrait_score": 84.0,
            "frontality_score": 80.0,
            "overall_score": 84.0,
            "embedding": emb_katy_2,
            "status": "kept",
        }
    )
    # Two detections from a completely different woman mistakenly inside Katy
    db.insert_detection(
        {
            "media_id": media["id"],
            "person_id": katy["id"],
            "portrait_score": 86.0,
            "frontality_score": 82.0,
            "overall_score": 86.0,
            "embedding": emb_other_1,
            "status": "kept",
        }
    )
    db.insert_detection(
        {
            "media_id": media["id"],
            "person_id": katy["id"],
            "portrait_score": 78.0,
            "frontality_score": 76.0,
            "overall_score": 78.0,
            "embedding": emb_other_2,
            "status": "kept",
        }
    )
    db.update_person_metadata(katy["id"], cover_detection_id=d1["id"])

    resp = client.post("/api/persons/recluster")
    assert resp.status_code == 200
    data = resp.json()
    assert data["persons_before"] == 1
    assert data["persons_after"] == 2
    assert data["reassigned_count"] == 2

    persons = client.get("/api/persons").json()
    names = {p["name"] for p in persons}
    assert "Katy" in names
    assert len(persons) == 2
    for p in persons:
        assert p["total_detections"] == 2


def test_clear_person_photos_preserves_identity_and_recognition(isolated_env):
    """Verify clearing all photos of a person removes images but keeps their profile & centroid alive for future matching."""
    from app.services.pipeline import match_or_create_person

    db = isolated_env["db"]
    dirs = isolated_env["dirs"]
    client = isolated_env["client"]

    emb = np.zeros(512, dtype=np.float32)
    emb[5] = 1.0

    portrait_file = dirs["PORTRAITS_DIR"] / "carlos_p1.jpg"
    portrait_file.write_bytes(b"portrait-bytes")

    media = db.create_media(
        media_id="m_carlos",
        original_filename="carlos.jpg",
        media_type="image",
        file_path="/tmp/carlos.jpg",
        file_size=100,
    )
    carlos = db.create_person("Carlos", centroid=emb)
    db.insert_detection(
        {
            "media_id": media["id"],
            "person_id": carlos["id"],
            "portrait_score": 90.0,
            "overall_score": 90.0,
            "portrait_path": portrait_file.name,
            "embedding": emb,
            "status": "kept",
        }
    )

    # Clear all photos while preserving Carlos
    res = client.post(f"/api/persons/{carlos['id']}/clear-photos")
    assert res.status_code == 200
    assert res.json() == {"cleared_detections": 1, "person_preserved": True}
    assert not portrait_file.exists()

    # Carlos remains in gallery with 0 photos and centroid intact
    detail = client.get(f"/api/persons/{carlos['id']}").json()
    assert detail["name"] == "Carlos"
    assert detail["total_detections"] == 0
    assert detail["cover_detection"] is None

    # Survives recluster even with 0 photos
    recluster = client.post("/api/persons/recluster")
    assert recluster.status_code == 200
    assert any(p["id"] == carlos["id"] and p["name"] == "Carlos" for p in client.get("/api/persons").json())

    # Future photo of Carlos matches his preserved centroid automatically!
    centroids = {pid: (name, vec, 3) for pid, name, vec in db.get_all_person_embeddings()}
    emb_future = np.zeros(512, dtype=np.float32)
    emb_future[5] = 0.98
    emb_future[6] = 0.19
    matched_id, matched_name, is_new = match_or_create_person(db, emb_future, 0.45, centroids)
    assert not is_new
    assert matched_id == carlos["id"]
    assert matched_name == "Carlos"


def test_merge_persons_preserves_photoless_centroid(isolated_env):
    """Verify merging a photo-less saved identity with another person preserves facial embeddings cleanly."""
    db = isolated_env["db"]
    client = isolated_env["client"]

    emb_a = np.zeros(512, dtype=np.float32)
    emb_a[10] = 1.0
    emb_b = np.zeros(512, dtype=np.float32)
    emb_b[11] = 1.0

    person_a = db.create_person("Elena", centroid=emb_a)
    person_b = db.create_person("Persona 2", centroid=emb_b)

    # Merge two photo-less identities -> centroid should combine both instead of becoming NULL
    res = client.post(
        f"/api/persons/{person_a['id']}/merge",
        json={"source_person_ids": [person_b["id"]]},
    )
    assert res.status_code == 200
    embs = {pid: vec for pid, _name, vec in db.get_all_person_embeddings()}
    assert person_a["id"] in embs
    assert person_b["id"] not in embs
    assert embs[person_a["id"]][10] > 0.5
    assert embs[person_a["id"]][11] > 0.5


def test_recluster_unites_same_person_and_enforces_cannot_link(isolated_env):
    """Verify recluster unites different-angle photos of the same person WITHOUT merging two faces in the same frame."""
    db = isolated_env["db"]
    client = isolated_env["client"]

    # Two photos of the SAME person with realistic pose/lighting variance (cosine sim ~0.44)
    emb_p1_front = np.zeros(512, dtype=np.float32)
    emb_p1_front[0] = 1.0
    emb_p1_angle = np.zeros(512, dtype=np.float32)
    emb_p1_angle[0] = 0.44
    emb_p1_angle[1] = float(np.sqrt(1.0 - 0.44 * 0.44))

    # A lookalike friend standing NEXT to them in photo 1 (same frame -> Cannot-Link!)
    emb_twin_same_frame = np.zeros(512, dtype=np.float32)
    emb_twin_same_frame[0] = 0.60
    emb_twin_same_frame[2] = float(np.sqrt(1.0 - 0.60 * 0.60))

    m1 = db.create_media(
        media_id="grp1",
        original_filename="group1.jpg",
        media_type="image",
        file_path="/tmp/group1.jpg",
        file_size=100,
    )
    m2 = db.create_media(
        media_id="solo2",
        original_filename="solo2.jpg",
        media_type="image",
        file_path="/tmp/solo2.jpg",
        file_size=100,
    )

    # Start fragmented as 3 separate unnamed persons
    pa = db.create_person("Persona 1", centroid=emb_p1_front)
    pb = db.create_person("Persona 2", centroid=emb_p1_angle)
    pc = db.create_person("Persona 3", centroid=emb_twin_same_frame)

    db.insert_detection(
        {
            "media_id": m1["id"],
            "person_id": pa["id"],
            "frame_idx": 0,
            "face_bbox": [50.0, 50.0, 150.0, 150.0],
            "portrait_score": 82.0,
            "frontality_score": 80.0,
            "overall_score": 82.0,
            "embedding": emb_p1_front,
            "status": "kept",
        }
    )
    db.insert_detection(
        {
            "media_id": m2["id"],
            "person_id": pb["id"],
            "frame_idx": 0,
            "face_bbox": [60.0, 60.0, 160.0, 160.0],
            "portrait_score": 40.0,  # moderate/candid angle (previously over-split!)
            "frontality_score": 38.0,
            "overall_score": 40.0,
            "embedding": emb_p1_angle,
            "status": "kept",
        }
    )
    db.insert_detection(
        {
            "media_id": m1["id"],
            "person_id": pc["id"],
            "frame_idx": 0,
            "face_bbox": [300.0, 50.0, 400.0, 150.0],  # same photo m1:0 -> Cannot-Link with pa!
            "portrait_score": 78.0,
            "frontality_score": 75.0,
            "overall_score": 78.0,
            "embedding": emb_twin_same_frame,
            "status": "kept",
        }
    )

    resp = client.post("/api/persons/recluster")
    assert resp.status_code == 200
    data = resp.json()
    # pa + pb unite into 1 person, while pc stays separate due to Cannot-Link in m1:0!
    assert data["persons_before"] == 3
    assert data["persons_after"] == 2
    counts = sorted(p["total_detections"] for p in client.get("/api/persons").json())
    assert counts == [1, 2]

