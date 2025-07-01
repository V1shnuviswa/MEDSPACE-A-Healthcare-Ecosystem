document.addEventListener("DOMContentLoaded", () => {
    const modeToggle = document.getElementById("toggle-auth");
    const formTitle = document.getElementById("form-title");
    const formSubtitle = document.getElementById("form-subtitle");
    const nameField = document.getElementById("name-field");
    const rememberSection = document.getElementById("remember-section");
    const submitButton = document.getElementById("submit-btn");
    let mode = "login";

    modeToggle.addEventListener("click", () => {
        if (mode === "login") {
            mode = "register";
            formTitle.textContent = "Create Account";
            formSubtitle.textContent = "Register for a new account";
            nameField.classList.remove("hidden");
            rememberSection.classList.add("hidden");
            submitButton.textContent = "Create Account";
            modeToggle.textContent = "Sign in";
        } else {
            mode = "login";
            formTitle.textContent = "Welcome Back";
            formSubtitle.textContent = "Sign in to your account";
            nameField.classList.add("hidden");
            rememberSection.classList.remove("hidden");
            submitButton.textContent = "Sign In";
            modeToggle.textContent = "Sign up";
        }
    });

    document.getElementById("auth-form").addEventListener("submit", async (e) => {
        e.preventDefault();

        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value.trim();
        const fullName = document.getElementById("full-name")?.value.trim();
        const userType = "patient"; // Removed doctor button, so default to patient

        const endpoint = mode === "login" ? "/api/login" : "/api/register";
        const requestData = { email, password, user_type: userType };

        if (mode === "register") {
            requestData.full_name = fullName;
        }

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestData),
            });

            const result = await response.json();

            if (response.ok) {
                alert(result.message);
                if (mode === "login") {
                    window.location.href = "/home";  // Redirect to home page after successful login
                }
            } else {
                alert(result.error);
            }
        } catch (error) {
            console.error("Error:", error);
            alert("Something went wrong. Please check the console.");
        }
    });
});
