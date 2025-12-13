module.exports = {
  apps: [
    {
      name: "camera-backend",
      script: "server.js",
      cwd: "/root/Forest-Demo/backend",
      env: {
        NODE_ENV: "production",
        PORT: 3003,
        AI_SERVER_URL: "http://localhost:8000/identify",
        BFF_CLIENT_LOOKUP_URL: "https://backend.chakhyudemo.com/api/traps/lookup",
        BFF_STORE_URL: "https://backend.chakhyudemo.com/api/detection",
        CLOUDINARY_CLOUD_NAME: "PUT_YOUR_CLOUD_NAME",
        CLOUDINARY_API_KEY: "PUT_YOUR_API_KEY",
        CLOUDINARY_API_SECRET: "PUT_YOUR_API_SECRET"
      }
    }
  ]
}
