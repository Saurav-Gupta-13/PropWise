export const metadata = {
  title: "PropWise — Mumbai Real Estate AI",
  description: "ML-powered fair market price predictions for Mumbai properties. 3 models trained on 5,053 real listings.",
  themeColor: "#0f172a",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "PropWise" },
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: "#0f172a",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet" />
      </head>
      <body style={{ margin: 0, padding: 0, fontFamily: "'Inter', system-ui, -apple-system, sans-serif", overflowX: "hidden" }}>
        {children}
      </body>
    </html>
  );
}
