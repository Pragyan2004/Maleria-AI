import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-analytics.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

const firebaseConfig = window.firebaseConfig;

let app;
let analytics;
let auth;

if (firebaseConfig && firebaseConfig.apiKey) {
    app = initializeApp(firebaseConfig);
    analytics = getAnalytics(app);
    auth = getAuth(app);
    console.log("Firebase services initialized");
} else {
    console.error("Firebase config missing");
}

export { app, auth, analytics };
