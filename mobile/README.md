# Nutri-AI Mobile (Expo SDK 57)

Native Expo SDK 57 client for the NutriGuide API. It is intentionally separate from the browser frontend: the phone experience is camera-first, uses a five-tab native layout, stores the session in the device secure store, and presents the analysis as a dark, high-contrast nutrition dashboard.

## Run

```sh
cp .env.example .env
npm install
npx expo start --lan
```

Set `EXPO_PUBLIC_API_URL` to the backend address visible from the simulator or phone. The backend must be started with `uvicorn main:app --host 0.0.0.0 --port 8000` when testing on a physical device.

The current local setup uses `exp://192.168.1.9:8081` for Metro and `http://192.168.1.9:8001` for the LAN API.
