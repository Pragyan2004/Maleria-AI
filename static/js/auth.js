import { auth } from './firebase-config.js';
import {
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    onAuthStateChanged,
    signOut,
    updateProfile,
    GoogleAuthProvider,
    signInWithPopup
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

// DOM Elements
const authSection = document.getElementById('authSection');
const userSection = document.getElementById('userSection');
const userProfilePic = document.getElementById('userProfilePic');
const userNameDis = document.getElementById('userNameDis');
const userEmailDis = document.getElementById('userEmailDis');
const logoutBtn = document.getElementById('logoutBtn');

const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const googleLoginBtn = document.getElementById('googleLogin');

// Listen for Auth State changes
onAuthStateChanged(auth, (user) => {
    if (user) {
        // User is signed in
        if (authSection) authSection.classList.add('d-none');
        if (userSection) {
            userSection.classList.remove('d-none');
            userSection.classList.add('d-flex');
        }

        if (userProfilePic) userProfilePic.src = user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName || 'User'}&background=0ea5e9&color=fff`;
        if (userNameDis) userNameDis.textContent = user.displayName || 'Researcher';
        if (userEmailDis) userEmailDis.textContent = user.email;

        // Redirect from login/register if already logged in
        if (window.location.pathname.includes('login') || window.location.pathname.includes('register')) {
            window.location.href = '/';
        }
    } else {
        // User is signed out
        if (authSection) {
            authSection.classList.remove('d-none');
            authSection.classList.add('d-flex');
        }
        if (userSection) {
            userSection.classList.add('d-none');
            userSection.classList.remove('d-flex');
        }
    }
});

// Handle Email/Password Login
if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        try {
            await signInWithEmailAndPassword(auth, email, password);
            window.location.href = '/';
        } catch (error) {
            alert("Login Failed: " + error.message);
        }
    });
}

// Handle Registration
if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const displayName = document.getElementById('displayName').value;

        try {
            const userCredential = await createUserWithEmailAndPassword(auth, email, password);
            await updateProfile(userCredential.user, {
                displayName: displayName
            });
            window.location.href = '/';
        } catch (error) {
            alert("Registration Failed: " + error.message);
        }
    });
}

// Handle Google Login
if (googleLoginBtn) {
    googleLoginBtn.addEventListener('click', async () => {
        const { GoogleAuthProvider, signInWithPopup } = await import("https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js");
        const provider = new GoogleAuthProvider();
        try {
            await signInWithPopup(auth, provider);
            window.location.href = '/';
        } catch (error) {
            alert("Google Auth Failed: " + error.message);
        }
    });
}

// Handle Logout
if (logoutBtn) {
    logoutBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        try {
            await signOut(auth);
            window.location.href = '/login';
        } catch (error) {
            console.error("Logout Error", error);
        }
    });
}
