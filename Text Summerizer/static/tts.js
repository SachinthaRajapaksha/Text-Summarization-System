// tts.js
let voices = [];

function loadVoices() {
    voices = speechSynthesis.getVoices();
    const voiceSelect = document.getElementById('voice-select');
    voiceSelect.innerHTML = '<option value="">Default Voice</option>';
    voices.forEach((voice, i) => {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `${voice.name} (${voice.lang})`;
        voiceSelect.appendChild(option);
    });
}

// Load voices when they're ready
if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = loadVoices;
}

function speakText(elementId) {
    const text = document.getElementById(elementId).innerText;
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Customize voice settings
    utterance.rate = 1; // Speed of speech (0.1 to 10)
    utterance.pitch = 1; // Pitch of voice (0 to 2)
    
    const voiceSelect = document.getElementById('voice-select');
    const selectedVoice = voiceSelect.value;
    
    if (selectedVoice !== "") {
        utterance.voice = voices[selectedVoice];
    }

    speechSynthesis.speak(utterance);
}

// Optional: Add a function to stop speaking
function stopSpeaking() {
    speechSynthesis.cancel();
}

// Load voices immediately in case they're already available
loadVoices();

// Event listener for voice selection change
document.getElementById('voice-select').addEventListener('change', function() {
    stopSpeaking(); // Stop any ongoing speech when voice is changed
});