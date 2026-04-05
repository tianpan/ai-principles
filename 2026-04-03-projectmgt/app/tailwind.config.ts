import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                "eng-blue": "#1677ff",
                "eng-orange": "#fa8c16",
                "eng-green": "#52c41a",
                "eng-red": "#ff4d4f",
                "eng-gray": {
                    50: "#fafafa",
                    100: "#f5f5f5",
                    200: "#f0f0f0",
                    300: "#d9d9d9",
                    400: "#bfbfbf",
                    500: "#8c8c8c",
                    600: "#595959",
                    700: "#434343",
                    800: "#262626",
                    900: "#1f1f1f",
                },
            },
            borderRadius: {
                "card": "8px",
                "btn": "4px",
            },
            maxWidth: {
                "app": "430px",
            },
            width: {
                "sidebar": "220px",
            },
            fontSize: {
                "app-title": ["18px", { lineHeight: "1.5" }],
                "app-body": ["14px", { lineHeight: "1.5" }],
            },
        },
    },
    plugins: [],
};
export default config;
