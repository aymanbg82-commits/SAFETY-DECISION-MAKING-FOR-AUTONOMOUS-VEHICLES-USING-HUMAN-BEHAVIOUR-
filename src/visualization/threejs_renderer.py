"""
Interactive Three.js 3D WebGL Simulation Renderer Component.
Generates self-contained HTML/JS bundle for embedding inside Streamlit or web browsers.
Features:
- Physically-scaled 3D road environment (asphalt, lane markings, curbs, sidewalks, streetlights)
- Procedural stylized 3D meshes (Ego car with active brake lights, motorcycle with rider, autorickshaw, pedestrian, truck)
- Trajectory visualization: past history ribbon, future predictions, 3D uncertainty confidence ellipses, replay ground truth
- Camera modes: Follow Camera, Overhead Bird's-Eye View, Free 3D Orbit
- Interactive playback HUD: Play/Pause, timeline scrubbing, speed control, overlay toggles, live telemetry
"""

import os
import json
from typing import Dict, Any, Optional


def render_threejs_simulation_html(
    payload: Dict[str, Any],
    height: int = 600,
    show_uncertainty_default: bool = True
) -> str:
    """
    Generate the complete, self-contained HTML page string containing Three.js 3D scene.
    """
    # Load local Three.js & OrbitControls scripts if present, otherwise fallback to CDN
    three_js_code = ""
    orbit_js_code = ""

    local_three = os.path.join(os.path.dirname(__file__), "..", "static", "three.min.js")
    local_orbit = os.path.join(os.path.dirname(__file__), "..", "static", "OrbitControls.js")

    if os.path.exists(local_three):
        with open(local_three, "r", encoding="utf-8") as f:
            three_js_code = f.read()

    if os.path.exists(local_orbit):
        with open(local_orbit, "r", encoding="utf-8") as f:
            orbit_js_code = f.read()

    payload_json = json.dumps(payload)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Traffic Safety Simulation Viewport</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            overflow: hidden;
            background-color: #0F172A;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #F8FAFC;
            user-select: none;
        }}
        #viewport-container {{
            position: relative;
            width: 100%;
            height: {height}px;
            background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        }}
        #webgl-canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        /* HUD Top Banner */
        .hud-banner {{
            position: absolute;
            top: 12px;
            left: 12px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #94A3B8;
            z-index: 10;
        }}
        .hud-banner span {{
            color: #38BDF8;
            font-weight: 700;
        }}
        /* HUD Camera Controls */
        .hud-camera {{
            position: absolute;
            top: 12px;
            right: 12px;
            display: flex;
            gap: 6px;
            z-index: 10;
        }}
        .hud-btn {{
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(226, 232, 240, 0.25);
            color: #F1F5F9;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .hud-btn:hover {{
            background: #2563EB;
            border-color: #3B82F6;
        }}
        .hud-btn.active {{
            background: #2563EB;
            border-color: #60A5FA;
            font-weight: 600;
        }}
        /* HUD Bottom Playback Bar */
        .hud-playback {{
            position: absolute;
            bottom: 12px;
            left: 12px;
            right: 12px;
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(226, 232, 240, 0.25);
            border-radius: 8px;
            padding: 10px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            z-index: 10;
        }}
        .timeline-row {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .timeline-slider {{
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: #334155;
            border-radius: 3px;
            outline: none;
            cursor: pointer;
        }}
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #38BDF8;
            cursor: pointer;
            border: 2px solid #FFFFFF;
            box-shadow: 0 0 6px rgba(56, 189, 248, 0.8);
        }}
        .time-label {{
            font-size: 12px;
            font-variant-numeric: tabular-nums;
            color: #E2E8F0;
            font-weight: 600;
            min-width: 55px;
        }}
        .controls-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .toggle-label {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
            color: #CBD5E1;
            cursor: pointer;
        }}
        .telemetry-chip {{
            font-size: 12px;
            font-weight: 500;
            padding: 3px 8px;
            border-radius: 4px;
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #F8FAFC;
        }}
        .telemetry-chip.braking {{
            background: rgba(239, 68, 68, 0.25);
            border-color: #EF4444;
            color: #FCA5A5;
        }}
        .telemetry-chip.safe {{
            background: rgba(16, 185, 129, 0.2);
            border-color: #10B981;
            color: #6EE7B7;
        }}
    </style>

    <!-- Embedded Three.js & Controls -->
    <script>
        {three_js_code if three_js_code else '// Fallback: Three.js will load from CDN'}
    </script>
    {'<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>' if not three_js_code else ''}
    <script>
        {orbit_js_code if orbit_js_code else '// Fallback: OrbitControls will load from CDN'}
    </script>
    {'<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>' if not orbit_js_code else ''}
</head>
<body>
    <div id="viewport-container">
        <!-- Banner -->
        <div class="hud-banner">
            <span>SYNTHETIC RESEARCH SIMULATION</span> — NOT A REAL-WORLD VEHICLE CONTROLLER<br>
            <span id="uncertainty-label">UNCERTAINTY: uncalibrated display spread</span>
        </div>

        <!-- Camera Toolbar -->
        <div class="hud-camera">
            <button id="cam-follow" class="hud-btn active">Follow Ego</button>
            <button id="cam-overhead" class="hud-btn">Overhead View</button>
            <button id="cam-orbit" class="hud-btn">Free Orbit</button>
            <button id="cam-reset" class="hud-btn">Reset</button>
        </div>

        <canvas id="webgl-canvas"></canvas>

        <!-- Playback HUD -->
        <div class="hud-playback">
            <div class="timeline-row">
                <button id="play-pause-btn" class="hud-btn" style="min-width: 70px;">▶ Play</button>
                <button id="restart-btn" class="hud-btn">↺ Reset</button>
                <input type="range" id="timeline" class="timeline-slider" min="0" max="100" value="0" step="1">
                <span id="time-display" class="time-label">0.00 s</span>
            </div>

            <div class="controls-row">
                <div class="control-group">
                    <label class="toggle-label"><input type="checkbox" id="toggle-history" checked> Past History</label>
                    <label class="toggle-label"><input type="checkbox" id="toggle-pred" checked> Forecast Path</label>
                    <label class="toggle-label"><input type="checkbox" id="toggle-unc" {'checked' if show_uncertainty_default else ''}> Uncertainty 90%</label>
                    <label class="toggle-label"><input type="checkbox" id="toggle-replay" checked> Actual Replay</label>
                </div>

                <div class="control-group">
                    <span style="font-size: 11px; color: #94A3B8;">Speed:</span>
                    <button class="hud-btn speed-btn" data-speed="0.5">0.5x</button>
                    <button class="hud-btn speed-btn active" data-speed="1.0">1.0x</button>
                    <button class="hud-btn speed-btn" data-speed="2.0">2.0x</button>
                </div>

                <div class="control-group">
                    <span id="chip-speed" class="telemetry-chip">Speed: 0.0 km/h</span>
                    <span id="chip-action" class="telemetry-chip safe">Cruising</span>
                    <span id="chip-dist" class="telemetry-chip">Sep: -- m</span>
                </div>
            </div>
        </div>
    </div>

    <!-- 3D Simulation Execution Script -->
    <script>
        const simData = {payload_json};
        const frames = simData.frames || [];
        const totalFrames = frames.length;
        const actorsMeta = {{}};
        (simData.actors || []).forEach(a => {{ actorsMeta[a.id] = a; }});

        let scene, camera, renderer, controls;
        let cameraMode = 'follow'; // 'follow', 'overhead', 'orbit'
        let followedActorId = 'ego';
        let currentFrameIdx = 0;
        let isPlaying = false;
        let playbackSpeed = 1.0;
        let lastTimestamp = 0;
        let accumulatedTime = 0;
        const dt = simData.scenario_config.time_step || 0.1;
        document.getElementById('uncertainty-label').textContent = 'UNCERTAINTY: ' + (simData.scenario_config.uncertainty_label || 'uncalibrated display spread');

        // Visual Object Collections
        const actorMeshes = {{}};
        const historyLines = {{}};
        const forecastLines = {{}};
        const uncertaintyEllipses = {{}};
        const replayLines = {{}};
        let hazardMarker = null;

        function initScene() {{
            const container = document.getElementById('viewport-container');
            const canvas = document.getElementById('webgl-canvas');

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0F172A);
            scene.fog = new THREE.Fog(0x0F172A, 50, 160);

            camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 300);

            renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: false }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2 - 0.05; // Do not go below ground

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.65);
            scene.add(ambientLight);

            const sunLight = new THREE.DirectionalLight(0xFFFBEB, 0.85);
            sunLight.position.set(20, 45, 25);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 1024;
            sunLight.shadow.mapSize.height = 1024;
            sunLight.shadow.camera.near = 0.5;
            sunLight.shadow.camera.far = 120;
            sunLight.shadow.camera.left = -40;
            sunLight.shadow.camera.right = 60;
            sunLight.shadow.camera.top = 25;
            sunLight.shadow.camera.bottom = -25;
            scene.add(sunLight);

            buildRoadEnvironment();
            buildActorMeshes();
            buildTrajectoryOverlays();

            setCameraMode('follow');
            updateSceneToFrame(0);

            window.addEventListener('resize', onWindowResize);
            setupUIEventListeners();

            requestAnimationFrame(animate);
        }}

        function buildRoadEnvironment() {{
            // Ground Grass Plane
            const groundGeo = new THREE.PlaneGeometry(200, 80);
            const groundMat = new THREE.MeshStandardMaterial({{ color: 0x1E293B, roughness: 0.9 }});
            const ground = new THREE.Mesh(groundGeo, groundMat);
            ground.rotation.x = -Math.PI / 2;
            ground.position.set(20, -0.05, 0);
            ground.receiveShadow = true;
            scene.add(ground);

            // Asphalt Road (Width 8m, Length 160m)
            const roadGeo = new THREE.PlaneGeometry(160, 8.4);
            const roadMat = new THREE.MeshStandardMaterial({{ color: 0x27272A, roughness: 0.85 }});
            const road = new THREE.Mesh(roadGeo, roadMat);
            road.rotation.x = -Math.PI / 2;
            road.position.set(20, 0.0, 0);
            road.receiveShadow = true;
            scene.add(road);

            // Centerline Dashes (Yellow)
            const dashCount = 30;
            for (let i = 0; i < dashCount; i++) {{
                const dashGeo = new THREE.PlaneGeometry(2.5, 0.18);
                const dashMat = new THREE.MeshBasicMaterial({{ color: 0xFACC15 }});
                const dash = new THREE.Mesh(dashGeo, dashMat);
                dash.rotation.x = -Math.PI / 2;
                dash.position.set(-50 + i * 5.2, 0.01, 0);
                scene.add(dash);
            }}

            // Road Edge Lines (Solid White)
            [-4.0, 4.0].forEach(zOffset => {{
                const lineGeo = new THREE.PlaneGeometry(160, 0.2);
                const lineMat = new THREE.MeshBasicMaterial({{ color: 0xF1F5F9 }});
                const line = new THREE.Mesh(lineGeo, lineMat);
                line.rotation.x = -Math.PI / 2;
                line.position.set(20, 0.01, zOffset);
                scene.add(line);
            }});

            // Curbs & Sidewalks
            [-5.5, 5.5].forEach(zOffset => {{
                const walkGeo = new THREE.BoxGeometry(160, 0.25, 2.5);
                const walkMat = new THREE.MeshStandardMaterial({{ color: 0x475569, roughness: 0.8 }});
                const sidewalk = new THREE.Mesh(walkGeo, walkMat);
                sidewalk.position.set(20, 0.1, zOffset);
                sidewalk.receiveShadow = true;
                scene.add(sidewalk);
            }});

            // Zebra Crossing at X = 25m
            for (let s = -3.4; s <= 3.4; s += 0.8) {{
                const stripeGeo = new THREE.PlaneGeometry(0.5, 2.8);
                const stripeMat = new THREE.MeshBasicMaterial({{ color: 0xF8FAFC }});
                const stripe = new THREE.Mesh(stripeGeo, stripeMat);
                stripe.rotation.x = -Math.PI / 2;
                stripe.position.set(25.0, 0.02, s);
                scene.add(stripe);
            }}

            // Hazard Pulsing Ring Marker
            const ringGeo = new THREE.RingGeometry(1.6, 2.2, 32);
            const ringMat = new THREE.MeshBasicMaterial({{ color: 0xEF4444, side: THREE.DoubleSide, transparent: true, opacity: 0.0 }});
            hazardMarker = new THREE.Mesh(ringGeo, ringMat);
            hazardMarker.rotation.x = -Math.PI / 2;
            hazardMarker.position.set(0, 0.04, 0);
            scene.add(hazardMarker);
        }}

        function createCarMesh(colorHex, isEgo = false) {{
            const carGroup = new THREE.Group();

            // Lower Body
            const bodyGeo = new THREE.BoxGeometry(4.5, 0.7, 2.0);
            const bodyMat = new THREE.MeshStandardMaterial({{ color: colorHex, metalness: 0.3, roughness: 0.4 }});
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            body.position.y = 0.55;
            body.castShadow = true;
            carGroup.add(body);

            // Cabin Roof
            const cabinGeo = new THREE.BoxGeometry(2.3, 0.65, 1.7);
            const cabinMat = new THREE.MeshStandardMaterial({{ color: 0x1E293B, roughness: 0.2 }});
            const cabin = new THREE.Mesh(cabinGeo, cabinMat);
            cabin.position.set(-0.3, 1.15, 0);
            cabin.castShadow = true;
            carGroup.add(cabin);

            // 4 Wheels
            const wheelGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.25, 16);
            const wheelMat = new THREE.MeshStandardMaterial({{ color: 0x09090B, roughness: 0.9 }});
            [
                [1.3, -1.0], [1.3, 1.0], [-1.3, -1.0], [-1.3, 1.0]
            ].forEach(([wx, wz]) => {{
                const wheel = new THREE.Mesh(wheelGeo, wheelMat);
                wheel.rotation.x = Math.PI / 2;
                wheel.position.set(wx, 0.35, wz);
                wheel.castShadow = true;
                carGroup.add(wheel);
            }});

            // Headlights
            const headGeo = new THREE.BoxGeometry(0.1, 0.15, 0.35);
            const headMat = new THREE.MeshBasicMaterial({{ color: 0xFEF08A }});
            const hl1 = new THREE.Mesh(headGeo, headMat);
            hl1.position.set(2.26, 0.65, -0.7);
            const hl2 = new THREE.Mesh(headGeo, headMat);
            hl2.position.set(2.26, 0.65, 0.7);
            carGroup.add(hl1);
            carGroup.add(hl2);

            // Taillights (Active brake lights)
            const tailGeo = new THREE.BoxGeometry(0.1, 0.15, 0.4);
            const tailMat = new THREE.MeshBasicMaterial({{ color: 0xEF4444 }});
            const tl1 = new THREE.Mesh(tailGeo, tailMat);
            tl1.position.set(-2.26, 0.65, -0.7);
            const tl2 = new THREE.Mesh(tailGeo, tailMat);
            tl2.position.set(-2.26, 0.65, 0.7);
            carGroup.add(tl1);
            carGroup.add(tl2);
            carGroup.userData.taillights = [tl1, tl2];

            return carGroup;
        }}

        function createMotorcycleMesh() {{
            const mcGroup = new THREE.Group();
            // Frame
            const frameGeo = new THREE.BoxGeometry(1.8, 0.5, 0.4);
            const frameMat = new THREE.MeshStandardMaterial({{ color: 0xEA580C, roughness: 0.4 }});
            const frame = new THREE.Mesh(frameGeo, frameMat);
            frame.position.y = 0.55;
            frame.castShadow = true;
            mcGroup.add(frame);

            // Wheels
            const wheelGeo = new THREE.CylinderGeometry(0.38, 0.38, 0.15, 16);
            const wheelMat = new THREE.MeshStandardMaterial({{ color: 0x18181B }});
            const fw = new THREE.Mesh(wheelGeo, wheelMat);
            fw.rotation.x = Math.PI / 2;
            fw.position.set(0.85, 0.38, 0);
            const rw = new THREE.Mesh(wheelGeo, wheelMat);
            rw.rotation.x = Math.PI / 2;
            rw.position.set(-0.85, 0.38, 0);
            mcGroup.add(fw);
            mcGroup.add(rw);

            // Rider Torso & Helmet
            const torsoGeo = new THREE.BoxGeometry(0.45, 0.6, 0.4);
            const torsoMat = new THREE.MeshStandardMaterial({{ color: 0x1E3A8A }});
            const torso = new THREE.Mesh(torsoGeo, torsoMat);
            torso.position.set(-0.1, 1.1, 0);
            torso.rotation.z = -0.15;
            mcGroup.add(torso);

            const helmetGeo = new THREE.SphereGeometry(0.2, 16, 16);
            const helmetMat = new THREE.MeshStandardMaterial({{ color: 0xFACC15 }});
            const helmet = new THREE.Mesh(helmetGeo, helmetMat);
            helmet.position.set(0.0, 1.5, 0);
            mcGroup.add(helmet);

            return mcGroup;
        }}

        function createAutorickshawMesh() {{
            const autoGroup = new THREE.Group();
            // Green Lower Body
            const bodyGeo = new THREE.BoxGeometry(2.8, 0.7, 1.3);
            const bodyMat = new THREE.MeshStandardMaterial({{ color: 0x15803D, roughness: 0.5 }});
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            body.position.y = 0.6;
            body.castShadow = true;
            autoGroup.add(body);

            // Yellow Canopy Roof
            const roofGeo = new THREE.BoxGeometry(2.0, 0.8, 1.35);
            const roofMat = new THREE.MeshStandardMaterial({{ color: 0xEAB308, roughness: 0.6 }});
            const roof = new THREE.Mesh(roofGeo, roofMat);
            roof.position.set(-0.2, 1.35, 0);
            roof.castShadow = true;
            autoGroup.add(roof);

            // 3 Wheels (1 front, 2 rear)
            const wGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.15, 16);
            const wMat = new THREE.MeshStandardMaterial({{ color: 0x18181B }});
            const fw = new THREE.Mesh(wGeo, wMat);
            fw.rotation.x = Math.PI / 2;
            fw.position.set(1.1, 0.28, 0);
            const rw1 = new THREE.Mesh(wGeo, wMat);
            rw1.rotation.x = Math.PI / 2;
            rw1.position.set(-0.9, 0.28, -0.65);
            const rw2 = new THREE.Mesh(wGeo, wMat);
            rw2.rotation.x = Math.PI / 2;
            rw2.position.set(-0.9, 0.28, 0.65);
            autoGroup.add(fw);
            autoGroup.add(rw1);
            autoGroup.add(rw2);

            return autoGroup;
        }}

        function createPedestrianMesh() {{
            const pedGroup = new THREE.Group();
            // Legs
            const legGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.75, 8);
            const legMat = new THREE.MeshStandardMaterial({{ color: 0x1E293B }});
            const l1 = new THREE.Mesh(legGeo, legMat);
            l1.position.set(0, 0.38, -0.15);
            const l2 = new THREE.Mesh(legGeo, legMat);
            l2.position.set(0, 0.38, 0.15);
            pedGroup.add(l1);
            pedGroup.add(l2);

            // Torso
            const torsoGeo = new THREE.BoxGeometry(0.35, 0.6, 0.45);
            const torsoMat = new THREE.MeshStandardMaterial({{ color: 0x16A34A }});
            const torso = new THREE.Mesh(torsoGeo, torsoMat);
            torso.position.y = 1.05;
            torso.castShadow = true;
            pedGroup.add(torso);

            // Head
            const headGeo = new THREE.SphereGeometry(0.16, 12, 12);
            const headMat = new THREE.MeshStandardMaterial({{ color: 0xFBCFE8 }});
            const head = new THREE.Mesh(headGeo, headMat);
            head.position.y = 1.5;
            head.castShadow = true;
            pedGroup.add(head);

            return pedGroup;
        }}

        function createAnimalMesh() {{
            const group = new THREE.Group();
            const body = new THREE.Mesh(
                new THREE.BoxGeometry(0.9, 0.65, 1.35),
                new THREE.MeshStandardMaterial({{ color: 0xA16207 }})
            );
            body.position.y = 0.45;
            body.castShadow = true;
            group.add(body);
            const head = new THREE.Mesh(
                new THREE.BoxGeometry(0.55, 0.5, 0.55),
                new THREE.MeshStandardMaterial({{ color: 0xD97706 }})
            );
            head.position.set(0.0, 0.7, -0.75);
            head.castShadow = true;
            group.add(head);
            return group;
        }}

        function buildActorMeshes() {{
            // Build Ego Car
            const egoMesh = createCarMesh(0x2563EB, true);
            scene.add(egoMesh);
            actorMeshes['ego'] = egoMesh;

            // Build Surrounding Actors
            Object.values(actorsMeta).forEach(meta => {{
                if (meta.is_ego) return;
                let mesh;
                if (meta.type === 'motorcycle') {{
                    mesh = createMotorcycleMesh();
                }} else if (meta.type === 'autorickshaw') {{
                    mesh = createAutorickshawMesh();
                }} else if (meta.type === 'pedestrian') {{
                    mesh = createPedestrianMesh();
                }} else if (meta.type === 'animal') {{
                    mesh = createAnimalMesh();
                }} else {{
                    mesh = createCarMesh(0x64748B, false);
                }}
                scene.add(mesh);
                actorMeshes[meta.id] = mesh;
            }});
        }}

        function buildTrajectoryOverlays() {{
            // Overlays are updated dynamically during frame updates
        }}

        function updateSceneToFrame(frameIdx) {{
            if (frameIdx < 0 || frameIdx >= totalFrames) return;
            currentFrameIdx = frameIdx;
            const f = frames[frameIdx];

            // 1. Update Ego Vehicle
            const egoMesh = actorMeshes['ego'];
            if (egoMesh) {{
                egoMesh.position.set(f.ego.x, f.ego.y, f.ego.z);
                egoMesh.rotation.y = f.ego.heading;
                // Update brake lights
                if (egoMesh.userData.taillights) {{
                    const bColor = f.ego.is_braking ? 0xFF0000 : 0x7F1D1D;
                    egoMesh.userData.taillights.forEach(tl => {{ tl.material.color.setHex(bColor); }});
                }}
            }}

            // 2. Update Surrounding Actors
            const currentActorIds = new Set();
            f.actors.forEach(a => {{
                currentActorIds.add(a.id);
                const mesh = actorMeshes[a.id];
                if (mesh) {{
                    mesh.visible = true;
                    mesh.position.set(a.x, a.y, a.z);
                    mesh.rotation.y = a.heading;
                }}
            }});

            // Hide actors not in this frame
            Object.keys(actorMeshes).forEach(aid => {{
                if (aid !== 'ego' && !currentActorIds.has(aid)) {{
                    actorMeshes[aid].visible = false;
                }}
            }});

            // 3. Update Camera in Follow Mode
            const followedMesh = actorMeshes[followedActorId] || egoMesh;
            if (cameraMode === 'follow' && followedMesh) {{
                camera.position.set(followedMesh.position.x - 14, 6.5, followedMesh.position.z);
                controls.target.set(followedMesh.position.x + 8, 1.0, followedMesh.position.z);
                camera.lookAt(controls.target);
            }}

            // 4. Update Hazard Ring
            if (hazardMarker) {{
                const sep = f.risk.min_separation;
                if (sep < 2.5) {{
                    hazardMarker.visible = true;
                    hazardMarker.position.set(f.ego.x + 2.0, 0.03, f.ego.z);
                    hazardMarker.material.opacity = Math.max(0.2, 1.0 - sep / 2.5);
                }} else {{
                    hazardMarker.visible = false;
                }}
            }}

            // 5. Update Forecast / Uncertainty Lines
            drawForecastOverlays(f);

            // 6. Update HUD Telemetry
            updateHUDTelemetry(f);
        }}

        function drawForecastOverlays(f) {{
            const showPred = document.getElementById('toggle-pred').checked;
            const showUnc = document.getElementById('toggle-unc').checked;
            const showHistory = document.getElementById('toggle-history').checked;
            const showReplay = document.getElementById('toggle-replay').checked;

            // Remove existing dynamic trajectory lines
            Object.values(forecastLines).forEach(line => scene.remove(line));
            Object.values(uncertaintyEllipses).forEach(mesh => scene.remove(mesh));
            Object.values(historyLines).forEach(line => scene.remove(line));
            Object.values(replayLines).forEach(line => scene.remove(line));

            if (showHistory || showReplay) {{
                Object.keys(actorMeshes).forEach(aid => {{
                    const points = frames.slice(0, currentFrameIdx + 1).map(frame => {{
                        const state = aid === 'ego' ? frame.ego : (frame.actors || []).find(actor => actor.id === aid);
                        return state ? new THREE.Vector3(state.x, 0.12, state.z) : null;
                    }}).filter(Boolean);
                    if (points.length < 2) return;
                    const line = new THREE.Line(
                        new THREE.BufferGeometry().setFromPoints(points),
                        new THREE.LineBasicMaterial({{ color: aid === 'ego' ? 0x38BDF8 : 0xCBD5E1 }})
                    );
                    scene.add(line);
                    if (showHistory) historyLines[aid] = line;
                    else scene.remove(line);
                }});
            }}
            if (showReplay) {{
                Object.keys(actorMeshes).forEach(aid => {{
                    const points = frames.map(frame => {{
                        const state = aid === 'ego' ? frame.ego : (frame.actors || []).find(actor => actor.id === aid);
                        return state ? new THREE.Vector3(state.x, 0.08, state.z) : null;
                    }}).filter(Boolean);
                    if (points.length < 2) return;
                    const line = new THREE.Line(
                        new THREE.BufferGeometry().setFromPoints(points),
                        new THREE.LineDashedMaterial({{ color: 0x94A3B8, dashSize: 0.35, gapSize: 0.2 }})
                    );
                    line.computeLineDistances();
                    scene.add(line);
                    replayLines[aid] = line;
                }});
            }}
            if (!showPred && !showUnc) return;

            Object.entries(f.predictions || {{}}).forEach(([aid, predData]) => {{
                const pts = predData.points;
                if (!pts || pts.length === 0) return;

                if (showPred) {{
                    const lineGeo = new THREE.BufferGeometry().setFromPoints(
                        pts.map(p => new THREE.Vector3(p[0], 0.1, p[2]))
                    );
                    const lineMat = new THREE.LineDashedMaterial({{
                        color: 0xF59E0B, dashSize: 0.8, gapSize: 0.4, linewidth: 2
                    }});
                    const line = new THREE.Line(lineGeo, lineMat);
                    line.computeLineDistances();
                    scene.add(line);
                    forecastLines[aid] = line;
                }}

                if (showUnc && predData.sigma_x) {{
                    // Draw uncertainty confidence boundary disk at horizon end
                    const endPt = pts[pts.length - 1];
                    const sx = predData.sigma_x[predData.sigma_x.length - 1] * 2.1;
                    const sy = predData.sigma_y[predData.sigma_y.length - 1] * 2.1;

                    const ellipseGeo = new THREE.CylinderGeometry(sx, sx, 0.08, 24);
                    ellipseGeo.scale(1, 1, sy / (sx || 1));
                    const ellipseMat = new THREE.MeshBasicMaterial({{
                        color: 0xEF4444, transparent: true, opacity: 0.25
                    }});
                    const ellMesh = new THREE.Mesh(ellipseGeo, ellipseMat);
                    ellMesh.position.set(endPt[0], 0.06, endPt[2]);
                    scene.add(ellMesh);
                    uncertaintyEllipses[aid] = ellMesh;
                }}
            }});
        }}

        function updateHUDTelemetry(f) {{
            document.getElementById('timeline').value = f.step;
            document.getElementById('time-display').textContent = f.timestamp.toFixed(2) + ' s';

            const spdKmh = (f.ego.speed * 3.6).toFixed(1);
            document.getElementById('chip-speed').textContent = `Speed: ${{spdKmh}} km/h`;

            const actionChip = document.getElementById('chip-action');
            actionChip.textContent = f.risk.action_type.replace('_', ' ').toUpperCase();
            if (f.risk.action_type === 'emergency_braking') {{
                actionChip.className = 'telemetry-chip braking';
            }} else {{
                actionChip.className = 'telemetry-chip safe';
            }}

            const sepVal = f.risk.min_separation < 50 ? `${{f.risk.min_separation.toFixed(1)}} m` : '> 50 m';
            document.getElementById('chip-dist').textContent = `Sep: ${{sepVal}}`;
        }}

        function setCameraMode(mode) {{
            cameraMode = mode;
            document.querySelectorAll('.hud-camera .hud-btn').forEach(b => b.classList.remove('active'));

            if (mode === 'follow') {{
                document.getElementById('cam-follow').classList.add('active');
                controls.enabled = false;
            }} else if (mode === 'overhead') {{
                document.getElementById('cam-overhead').classList.add('active');
                controls.enabled = false;
                const ego = frames[currentFrameIdx] ? frames[currentFrameIdx].ego : {{ x: 20, z: 0 }};
                camera.position.set(ego.x + 5, 36, 0);
                camera.lookAt(ego.x + 5, 0, 0);
            }} else if (mode === 'orbit') {{
                document.getElementById('cam-orbit').classList.add('active');
                controls.enabled = true;
            }}
        }}

        function setupUIEventListeners() {{
            const playBtn = document.getElementById('play-pause-btn');
            playBtn.addEventListener('click', () => {{
                isPlaying = !isPlaying;
                playBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            }});

            document.getElementById('restart-btn').addEventListener('click', () => {{
                isPlaying = false;
                playBtn.textContent = '▶ Play';
                updateSceneToFrame(0);
            }});

            const timeline = document.getElementById('timeline');
            timeline.max = totalFrames - 1;
            timeline.addEventListener('input', (e) => {{
                isPlaying = false;
                playBtn.textContent = '▶ Play';
                updateSceneToFrame(parseInt(e.target.value));
            }});

            document.getElementById('cam-follow').addEventListener('click', () => setCameraMode('follow'));
            document.getElementById('cam-overhead').addEventListener('click', () => setCameraMode('overhead'));
            document.getElementById('cam-orbit').addEventListener('click', () => setCameraMode('orbit'));
            document.getElementById('cam-reset').addEventListener('click', () => {{
                followedActorId = 'ego';
                setCameraMode('follow');
                updateSceneToFrame(currentFrameIdx);
            }});

            document.querySelectorAll('.speed-btn').forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    playbackSpeed = parseFloat(e.target.getAttribute('data-speed'));
                }});
            }});

            ['toggle-pred', 'toggle-unc', 'toggle-history', 'toggle-replay'].forEach(id => {{
                document.getElementById(id).addEventListener('change', () => {{
                    updateSceneToFrame(currentFrameIdx);
                }});
            }});
        }}

        function onWindowResize() {{
            const container = document.getElementById('viewport-container');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }}

        function animate(timestamp) {{
            requestAnimationFrame(animate);

            if (!lastTimestamp) lastTimestamp = timestamp;
            const deltaMs = timestamp - lastTimestamp;
            lastTimestamp = timestamp;

            if (isPlaying && totalFrames > 0) {{
                accumulatedTime += (deltaMs / 1000.0) * playbackSpeed;
                const frameInterval = dt;

                if (accumulatedTime >= frameInterval) {{
                    const stepsToAdvance = Math.floor(accumulatedTime / frameInterval);
                    accumulatedTime %= frameInterval;

                    let nextIdx = currentFrameIdx + stepsToAdvance;
                    if (nextIdx >= totalFrames) {{
                        nextIdx = totalFrames - 1;
                        isPlaying = false;
                        document.getElementById('play-pause-btn').textContent = '▶ Play';
                    }}
                    updateSceneToFrame(nextIdx);
                }}
            }}

            if (controls && controls.enabled) {{
                controls.update();
            }}

            renderer.render(scene, camera);
        }}

        window.onload = initScene;
    </script>
</body>
</html>
"""
    return html_template
