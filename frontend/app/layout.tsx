import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FlyGPT v0.7",
  description: "FlyGPT live FlyGraph streaming and compressed memory",
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
