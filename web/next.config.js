/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    ML_SERVICE_URL: process.env.ML_SERVICE_URL || "http://localhost:8000",
  },
};

module.exports = nextConfig;
