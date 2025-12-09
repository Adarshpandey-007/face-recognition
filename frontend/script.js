function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = type;
        
        // Clear after 5 seconds
        setTimeout(() => {
            statusDiv.textContent = '';
            statusDiv.className = '';
        }, 5000);
    }
}

async function initCamera() {
    const video = document.getElementById('video');
    if (video) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            showStatus("Error accessing webcam. Please allow camera access.", "error");
        }
    }
}
