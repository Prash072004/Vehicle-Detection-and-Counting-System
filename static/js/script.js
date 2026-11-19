/* One MJPEG element, serialized status polling, and server-confirmed controls. */
(function (root) {
    "use strict";
    const STATES = ["not_started", "initializing", "ready", "processing", "completed", "stopping", "stopped", "error"];
    const RECOGNITION = {
        not_attempted: "Not attempted", pending: "Recognition pending", unreadable: "Plate unreadable",
        accepted: "OCR accepted", conflicting: "Recognition uncertain", low_confidence: "Low confidence",
        format_unrecognized: "Format unrecognized"
    };
    const LABELS = {
        not_started: "Not started", initializing: "Initializing", ready: "Ready",
        processing: "Processing", completed: "Completed", stopping: "Stopping", stopped: "Stopped", error: "Error"
    };
    const number = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
    function validateStatus(data) {
        if (!data || data.schema_version !== 1 || !STATES.includes(data.state) ||
            !data.signal || !["red", "green"].includes(data.signal.color) ||
            !Number.isInteger(data.signal.version) || data.signal.version < 0 ||
            !data.source || !["unavailable", "local_video", "camera"].includes(data.source.kind) ||
            !data.metrics || !data.storage || !["unavailable", "disabled", "ready", "degraded"].includes(data.storage.state) ||
            !Array.isArray(data.recent_candidates) || data.recent_candidates.length > 20 ||
            (data.session != null && typeof data.session !== "string")) throw new Error("Invalid monitoring response");
        for (const key of ["processed_frames", "frame_index", "processing_fps", "source_fps", "vehicle_detections",
            "observed_tracks", "candidates", "accepted_plates", "active_viewers"]) {
            if (data.metrics[key] != null && !number(data.metrics[key])) throw new Error("Invalid monitoring metric");
        }
        for (const key of ["width", "height", "total_frames"]) {
            if (data.source[key] != null && !number(data.source[key])) throw new Error("Invalid source metric");
        }
        for (const row of data.recent_candidates) {
            if (!row || !Number.isInteger(row.track_id) || row.track_id < 1 ||
                !Number.isInteger(row.frame_index) || row.frame_index < 0 ||
                !["red", "green", "unknown"].includes(row.signal) ||
                !["candidate", "uncertain"].includes(row.status) || !Object.hasOwnProperty.call(RECOGNITION, row.recognition))
                throw new Error("Invalid candidate summary");
        }
        return data;
    }
    function validatePlates(data) {
        if (!Array.isArray(data) || !data.every(p => typeof p === "string" && /^[A-Z0-9]{1,32}$/.test(p)))
            throw new Error("Invalid plate response");
        return data;
    }
    function createClient(fetcher, timeout = 8000) {
        const controllers = new Set();
        return {
            async request(path, options = {}, json = true) {
                const controller = new AbortController();
                controllers.add(controller);
                const timer = setTimeout(() => controller.abort(), timeout);
                try {
                    const response = await fetcher(path, {...options, signal: controller.signal, cache: "no-store"});
                    if (!response.ok) throw new Error("Request failed");
                    return json ? await response.json() : response;
                } finally {
                    clearTimeout(timer);
                    controllers.delete(controller);
                }
            },
            abort() { for (const controller of controllers) controller.abort(); }
        };
    }
    function mount(doc, win, client = createClient(win.fetch.bind(win))) {
        const get = id => doc.getElementById(id);
        const text = (id, value) => { const el = get(id); if (el.textContent !== String(value)) el.textContent = value; };
        const forms = Array.from(doc.querySelectorAll("form[data-color]"));
        const stream = get("video-stream");
        let last = null, signalVersion = -1, busy = false, running = false, disposed = false;
        let timer = null, revision = 0, streamStarted = false, streamFailed = false, retry = 0;
        let candidatesKey = "", platesKey = "", platesAt = 0, platesSession = null, stale = true;
        const listeners = [];
        function listen(target, name, callback) { target.addEventListener(name, callback); listeners.push([target, name, callback]); }
        function buttons() {
            for (const form of forms) {
                const button = form.querySelector("button");
                button.disabled = busy || stale || Boolean(last && last.signal.color === form.dataset.color);
                button.setAttribute("aria-pressed", String(Boolean(last && last.signal.color === form.dataset.color)));
            }
        }
        function schedule() {
            clearTimeout(timer);
            if (!disposed && !doc.hidden && !busy) timer = setTimeout(poll, 2000);
        }
        function renderCandidates(data) {
            const key = JSON.stringify(data.recent_candidates);
            if (key !== candidatesKey) {
                candidatesKey = key;
                const fragment = doc.createDocumentFragment();
                for (const row of data.recent_candidates) {
                    const tr = doc.createElement("tr");
                    for (const value of ["#" + row.track_id, row.frame_index, row.signal.toUpperCase(),
                        row.status === "candidate" ? "Candidate" : "Uncertain candidate", RECOGNITION[row.recognition]]) {
                        const td = doc.createElement("td"); td.textContent = value; tr.appendChild(td);
                    }
                    fragment.appendChild(tr);
                }
                get("candidate-rows").replaceChildren(fragment);
            }
            const empty = data.recent_candidates.length === 0;
            get("candidates-empty").hidden = !empty;
            get("candidate-table-wrap").hidden = empty;
            text("candidates-empty", data.metrics.processed_frames == null || data.metrics.processed_frames === 0 ?
                (data.state === "error" ? "Processing is unavailable. No candidate data is available." : "Processing has not started.") :
                "No crossing candidates detected in this session.");
        }
        function renderVideo(data) {
            const active = ["ready", "processing"].includes(data.state);
            if (!streamStarted && active) {
                streamStarted = true;
                stream.src = stream.dataset.src;
            }
            const messages = {
                not_started: "Processing has not started.", initializing: "Initializing the video source and models…",
                ready: "Source ready. Waiting for the first processed frame…",
                completed: "Video completed. The last received frame may remain visible.",
                stopping: "Processing is stopping…", stopped: "Processing stopped.",
                error: "Processing unavailable. Check the source and server logs."
            };
            if (data.state === "processing") {
                get("video-message").hidden = !streamFailed;
                text("video-description", "Processing source · one shared session");
                if (streamFailed) text("video-message", "Stream disconnected. Processing may continue for other viewers.");
            } else {
                get("video-message").hidden = false;
                text("video-message", messages[data.state]);
                text("video-description", LABELS[data.state] + " · one shared session");
            }
            get("retry-stream").hidden = !(active && streamFailed);
        }
        function render(data) {
            // Old poll/signal responses must not overwrite a newer confirmed signal.
            if (data.signal.version < signalVersion && last && data.session === last.session) return;
            signalVersion = data.signal.version;
            last = data; stale = false;
            text("status-badge", LABELS[data.state]); get("status-badge").dataset.state = data.state;
            text("last-update", "Updated " + new Date().toLocaleTimeString());
            get("connection-message").hidden = true;
            text("current-signal", data.signal.color.toUpperCase());
            get("red-lamp").classList.toggle("active", data.signal.color === "red");
            get("green-lamp").classList.toggle("active", data.signal.color === "green");
            text("source-kind", {unavailable: "Source unavailable", local_video: "Local video", camera: "Camera input"}[data.source.kind]);
            text("session-id", data.session ? data.session.slice(0, 8) : "Unavailable");
            text("frame-size", data.source.width && data.source.height ? data.source.width + " × " + data.source.height : "Unavailable");
            const format = (value, decimal = false) => value == null ? "—" : decimal ? value.toFixed(1) : value.toLocaleString();
            for (const [id, key, decimal] of [
                ["metric-frames", "processed_frames"], ["metric-fps", "processing_fps", true],
                ["metric-detections", "vehicle_detections"], ["metric-tracks", "observed_tracks"],
                ["metric-candidates", "candidates"], ["metric-plates", "accepted_plates"]]) text(id, format(data.metrics[key], decimal));
            text("source-fps", data.metrics.source_fps == null ? "Unavailable" : format(data.metrics.source_fps, true) + " fps");
            text("active-viewers", format(data.metrics.active_viewers));
            text("storage-state", {unavailable:"Unavailable", disabled:"Disabled", ready:"Ready", degraded:"Degraded · check logs"}[data.storage.state]);
            text("evidence-status", (data.storage.evidence_capture ? "Evidence capture enabled. " : "Evidence capture disabled. ") +
                "Real evidence access requires an authorized review workflow.");
            const total = data.source.total_frames, index = data.metrics.frame_index;
            if (total > 0 && index != null) {
                const progress = Math.min(100, (index + 1) / total * 100);
                get("source-progress").value = progress;
                text("progress-text", Math.floor(progress) + "% · source frames reached");
            } else { get("source-progress").removeAttribute("value"); text("progress-text", "Unavailable"); }
            renderCandidates(data); renderVideo(data); buttons();
        }
        async function plates(data, token) {
            const accepted = data.metrics.accepted_plates;
            if (platesSession !== data.session) { platesAt = 0; platesKey = ""; get("violations-list").replaceChildren(); platesSession = data.session; }
            if (accepted == null || accepted === 0) {
                get("violations-list").replaceChildren(); platesKey = ""; platesAt = 0;
                text("plates-message", accepted == null ? "Plate results unavailable." :
                    data.metrics.candidates > 0 ? "Candidates detected; no accepted plate results." : "No accepted plate results.");
                return;
            }
            if (Date.now() - platesAt < 10000 && get("violations-list").childElementCount === accepted) return;
            try {
                const values = validatePlates(await client.request("/violations"));
                if (disposed || token !== revision) return;
                const key = JSON.stringify(values);
                if (platesKey !== key) {
                    platesKey = key;
                    const fragment = doc.createDocumentFragment();
                    // The compatibility endpoint is unchanged; DOM rendering is bounded.
                    for (const plate of values.slice(-100)) { const li = doc.createElement("li"); li.textContent = plate; fragment.appendChild(li); }
                    get("violations-list").replaceChildren(fragment);
                }
                platesAt = Date.now();
                text("plates-message", values.length ? "Current-session OCR results · showing up to 100" : "No accepted plate results.");
            } catch (error) {
                if (token === revision && !disposed) {
                    get("violations-list").replaceChildren(); platesAt = 0; platesKey = "";
                    text("plates-message", "Plate results unavailable. Retrying on the next monitoring update.");
                }
            }
        }
        async function poll() {
            if (disposed || doc.hidden || busy || running) return;
            running = true; const token = revision;
            try {
                const data = validateStatus(await client.request("/api/status"));
                if (disposed || token !== revision) return;
                render(data); await plates(data, token);
            } catch (error) {
                if (!disposed && token === revision) {
                    stale = true; buttons();
                    text("status-badge", "Connection unavailable"); get("status-badge").dataset.state = "error";
                    text("last-update", "Values are stale");
                    text("connection-message", "Monitoring updates unavailable. Displayed values are from the last successful update; retrying shortly.");
                    get("connection-message").hidden = false;
                }
            } finally {
                running = false;
                if (token === revision) schedule();
            }
        }
        async function setSignal(event, form) {
            event.preventDefault();
            if (busy || stale || disposed) return;
            busy = true; revision += 1; const token = revision;
            clearTimeout(timer); client.abort(); buttons();
            get("signal-feedback").classList.remove("error");
            text("signal-feedback", "Updating signal…");
            try {
                await client.request(form.action, {method: "POST"}, false);
                const data = validateStatus(await client.request("/api/status"));
                if (disposed || token !== revision) return;
                render(data);
                text("signal-feedback", "Server confirmed " + data.signal.color.toUpperCase() + ". Manual signal only.");
            } catch (error) {
                if (token === revision && !disposed) {
                    stale = true;
                    get("signal-feedback").classList.add("error");
                    text("signal-feedback", "Could not confirm the update. Last confirmed signal is shown; waiting for monitoring.");
                }
            } finally {
                if (!disposed && token !== revision) text("signal-feedback", "Update confirmation interrupted. Monitoring will refresh the signal.");
                busy = false; buttons(); schedule();
            }
        }
        for (const form of forms) listen(form, "submit", event => { void setSignal(event, form); });
        listen(stream, "error", () => {
            if (last && ["completed", "stopped", "error"].includes(last.state)) return;
            streamFailed = true;
            text("video-message", "Stream disconnected. Processing may continue for other viewers.");
            get("video-message").hidden = false;
            get("retry-stream").hidden = false;
        });
        listen(stream, "load", () => { streamFailed = false; if (last) renderVideo(last); });
        listen(get("retry-stream"), "click", () => {
            if (!last || !["ready", "processing"].includes(last.state)) return;
            streamFailed = false; get("retry-stream").hidden = true;
            text("video-message", "Reconnecting to the shared stream…");
            stream.src = stream.dataset.src + "?retry=" + (++retry);
        });
        function evidence() {
            const missing = get("evidence-kind").value === "missing";
            get("evidence-image").hidden = missing;
            if (missing) get("evidence-image").removeAttribute("src");
            else get("evidence-image").src = "/static/evidence-fixture.svg";
            text("evidence-caption", missing ? "Plate crop unavailable in this synthetic fixture." : "Synthetic crossing fixture · frame 42. Not real evidence.");
        }
        listen(get("evidence-demo"), "click", () => { evidence(); get("evidence-dialog").showModal(); });
        listen(get("close-evidence"), "click", () => get("evidence-dialog").close());
        listen(get("evidence-dialog"), "close", () => get("evidence-demo").focus());
        listen(get("evidence-kind"), "change", evidence);
        listen(get("evidence-image"), "error", () => {
            get("evidence-image").hidden = true; text("evidence-caption", "Synthetic preview could not be loaded.");
        });
        listen(doc, "visibilitychange", () => {
            if (doc.hidden) { revision += 1; clearTimeout(timer); client.abort(); }
            else { schedule(); }
        });
        function dispose() {
            disposed = true; revision += 1; clearTimeout(timer); client.abort();
            for (const [target, name, callback] of listeners) target.removeEventListener(name, callback);
        }
        listen(win, "pagehide", event => { if (!event.persisted) dispose(); else { revision += 1; clearTimeout(timer); client.abort(); } });
        listen(win, "pageshow", event => { if (event.persisted) schedule(); });
        buttons(); void poll();
        return {dispose, poll, render, setSignal};
    }
    const api = {validateStatus, validatePlates, createClient, mount, LABELS, RECOGNITION};
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    else { root.TrafficDashboard = api; mount(root.document, root); }
})(typeof window !== "undefined" ? window : globalThis);
