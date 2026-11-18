async function fetchViolations() {
    try {
        const response = await fetch("/violations");
        const data = await response.json();

        const list = document.getElementById("violations-list");
        list.innerHTML = "";

        if (data.length === 0) {
            list.innerHTML = "<li>No violations detected</li>";
        } else {
            data.forEach(plate => {
                const li = document.createElement("li");
                li.textContent = plate;
                list.appendChild(li);
            });
        }
    } catch (error) {
        console.error("Error fetching violations:", error);
    }
}

// Refresh every ~500ms (fast enough to match your YOLO frame updates)
setInterval(fetchViolations, 500);
fetchViolations();
