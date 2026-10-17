import Script from "next/script";
import type { Metadata } from "next";
import "./globals.css";
import "./refinements.css";
import "./liquid-refraction.css";

export const metadata: Metadata = {
  title: "FlyGPT v0.7.1",
  description: "Connectome-routed AI workspace",
  icons: {
    icon: "/flygpt-logo.webp",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}  <Script src="/liquid-refraction.js" strategy="afterInteractive" />
      </body>
    </html>
  );
}
