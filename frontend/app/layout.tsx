import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FlyGPT v0.7.1",
  description: "Connectome-routed AI workspace",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
