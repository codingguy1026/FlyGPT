import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FlyGPT v0.6",
  description: "FlyWire connectome router experiment",
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
