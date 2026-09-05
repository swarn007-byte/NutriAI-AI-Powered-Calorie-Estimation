import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Animated,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StatusBar as RNStatusBar,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Camera, CameraView } from "expo-camera";
import * as FileSystem from "expo-file-system/legacy";
import * as SecureStore from "expo-secure-store";
import { Ionicons } from "@expo/vector-icons";
import { StatusBar } from "expo-status-bar";
import Svg, { Circle, G, Line, Rect, Text as SvgText } from "react-native-svg";

const API_URL = (process.env.EXPO_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const TOKEN_KEY = "nutriai.mobile.token";
const USER_KEY = "nutriai.mobile.user";
const ONBOARD_KEY = "nutriai.mobile.onboarded";

const C = {
  bg: "#F0F0EE",
  card: "#FFFFFF",
  ink: "#111310",
  ink2: "#5F635A",
  muted: "#9A9D94",
  faint: "#BEC1B8",
  line: "#E6E7E1",
  lineSoft: "#F1F2ED",
  mint: "#D8F187",
  mintSoft: "#EDF8CC",
  green: "#8FD03F",
  greenDeep: "#5C9418",
  onGreen: "#14210B",
  dark: "#191B17",
  amber: "#F0A32B",
  blue: "#4A9DF0",
  pink: "#EE6E9B",
  danger: "#D9534A",
};

const SHADOW = Platform.select({
  ios: { shadowColor: "#1c1f18", shadowOpacity: 0.07, shadowRadius: 16, shadowOffset: { width: 0, height: 8 } },
  android: { elevation: 2 },
  default: {},
});

const SHADOW_SOFT = Platform.select({
  ios: { shadowColor: "#1c1f18", shadowOpacity: 0.05, shadowRadius: 8, shadowOffset: { width: 0, height: 3 } },
  android: { elevation: 1 },
  default: {},
});

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// Android draws edge-to-edge on SDK 54+ and react-native's SafeAreaView is a
// no-op there, so the system bar gaps have to be added by hand.
const TOP_INSET = Platform.OS === "android" ? RNStatusBar.currentHeight || 26 : 0;
const BOTTOM_INSET = Platform.OS === "android" ? 30 : 0;

const formatKcal = (value) => `${Math.round(Number(value) || 0)}`;
const formatGrams = (value) => `${Math.round(Number(value) || 0)} g`;
const percent = (value, goal) => {
  const target = Number(goal) || 0;
  if (target <= 0) return 0;
  return Math.max(0, Math.min(1, (Number(value) || 0) / target));
};
const titleCase = (value) => String(value || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const initialOf = (user) => (user?.name || user?.email || "G").trim().charAt(0).toUpperCase();
const greetingForHour = (hour) => (hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening");
const imageUrl = (path) => (!path ? null : path.startsWith("http") ? path : `${API_URL}${path}`);

const dayKey = (value) => {
  const d = parseDate(value);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

// The API serialises UTC timestamps without a designator, so an unmarked string
// must be forced to UTC or every meal drifts by the device's offset.
function parseDate(value) {
  if (value instanceof Date) return value;
  const text = String(value || "");
  const marked = /[zZ]$|[+-]\d\d:?\d\d$/.test(text);
  return new Date(marked ? text : `${text.replace(" ", "T")}Z`);
}

function weekDays(offsetWeeks) {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const monday = (start.getDay() + 6) % 7;
  start.setDate(start.getDate() - monday + offsetWeeks * 7);
  return Array.from({ length: 7 }, (_, index) => {
    const day = new Date(start);
    day.setDate(start.getDate() + index);
    return day;
  });
}

function detailText(payload, fallback) {
  const detail = payload && payload.detail;
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0] || {};
    if (typeof first.msg === "string" && first.msg) {
      const loc = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : null;
      return loc && loc !== "body" ? `${titleCase(loc)}: ${first.msg}` : first.msg;
    }
  }
  return fallback;
}

async function request(path, options = {}, token) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API_URL}${path}`, { ...options, headers });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const failure = new Error(detailText(payload, `Something went wrong (HTTP ${response.status}).`));
    failure.status = response.status;
    throw failure;
  }
  return payload;
}

async function uploadMealImage(asset, token, plateDiameter) {
  const info = await FileSystem.getInfoAsync(asset.uri, { size: true });
  if (!info.exists) throw new Error("That photo is no longer available on this device.");
  const response = await FileSystem.uploadAsync(`${API_URL}/api/meals/analyze`, asset.uri, {
    httpMethod: "POST",
    uploadType: FileSystem.FileSystemUploadType.MULTIPART,
    fieldName: "image",
    mimeType: asset.mimeType || "image/jpeg",
    parameters: { plate_diameter_cm: String(plateDiameter || 26) },
    headers: { Accept: "application/json", Authorization: `Bearer ${token}` },
  });
  let payload = null;
  try {
    payload = JSON.parse(response.body);
  } catch (reason) {
    payload = null;
  }
  if (response.status < 200 || response.status >= 300) {
    throw new Error(detailText(payload, `Upload failed (HTTP ${response.status}).`));
  }
  if (!payload) throw new Error("The server sent back a response we could not read.");
  return payload;
}

async function persistSession(token, user) {
  await SecureStore.setItemAsync(TOKEN_KEY, token);
  await SecureStore.setItemAsync(USER_KEY, JSON.stringify(user || {}));
}

async function clearSession() {
  await SecureStore.deleteItemAsync(TOKEN_KEY);
  await SecureStore.deleteItemAsync(USER_KEY);
}

async function readStoredSession() {
  const token = await SecureStore.getItemAsync(TOKEN_KEY);
  if (!token) return null;
  let user = null;
  try {
    user = JSON.parse((await SecureStore.getItemAsync(USER_KEY)) || "null");
  } catch (reason) {
    user = null;
  }
  return { token, user };
}

export default function App() {
  const [phase, setPhase] = useState("booting");
  const [session, setSession] = useState(null);
  const [screen, setScreen] = useState("home");
  const [meal, setMeal] = useState(null);
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState(null);
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [correction, setCorrection] = useState(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [upgrading, setUpgrading] = useState(false);

  const token = session?.token;
  const user = session?.user;
  const plateDiameter = user?.preferences?.plate_diameter_cm || 26;

  const loadDashboardData = useCallback(async (activeToken) => {
    const offset = -new Date().getTimezoneOffset();
    const [historyOut, summaryOut] = await Promise.all([
      request("/api/users/me/history?limit=60&offset=0", {}, activeToken),
      request(`/api/users/me/summary?days=14&tz_offset=${offset}`, {}, activeToken),
    ]);
    setHistory(historyOut?.meals || []);
    setSummary(summaryOut || null);
  }, []);

  const enterApp = useCallback(
    async (nextSession) => {
      await persistSession(nextSession.token, nextSession.user);
      setSession(nextSession);
      setScreen("home");
      setPhase("app");
      setUpgrading(false);
      setError(null);
      try {
        await loadDashboardData(nextSession.token);
      } catch (reason) {
        setError(reason.message);
      }
    },
    [loadDashboardData],
  );

  const signOut = useCallback(async () => {
    await clearSession();
    setSession(null);
    setHistory([]);
    setSummary(null);
    setMeal(null);
    setError(null);
    setScreen("home");
    setPhase("auth");
    setUpgrading(false);
  }, []);

  const beginUpgrade = useCallback(() => {
    setUpgrading(true);
    setPhase("auth");
  }, []);

  const cancelUpgrade = useCallback(() => {
    setUpgrading(false);
    setPhase("app");
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [seenOnboarding, stored] = await Promise.all([
        SecureStore.getItemAsync(ONBOARD_KEY),
        readStoredSession(),
      ]);
      if (cancelled) return;
      if (!stored) {
        setPhase(seenOnboarding ? "auth" : "onboarding");
        return;
      }
      setSession(stored);
      setPhase("app");
      try {
        const fresh = await request("/api/auth/me", {}, stored.token);
        if (cancelled) return;
        setSession({ token: stored.token, user: fresh });
        await persistSession(stored.token, fresh);
        await loadDashboardData(stored.token);
      } catch (reason) {
        if (cancelled) return;
        if (reason.status === 401 || reason.status === 403) {
          await clearSession();
          setSession(null);
          setPhase("auth");
          return;
        }
        setError(reason.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadDashboardData]);

  const finishOnboarding = useCallback(async () => {
    await SecureStore.setItemAsync(ONBOARD_KEY, "1");
    setPhase("auth");
  }, []);

  const refresh = useCallback(async () => {
    if (!token) return;
    setRefreshing(true);
    try {
      await loadDashboardData(token);
      setError(null);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setRefreshing(false);
    }
  }, [loadDashboardData, token]);

  const analyse = useCallback(
    async (asset) => {
      if (!asset || !token) return;
      setBusy(true);
      setError(null);
      try {
        const result = await uploadMealImage(asset, token, plateDiameter);
        setMeal(result);
        setScreen("results");
        await loadDashboardData(token);
      } catch (reason) {
        console.error("[analyse] upload failed:", reason);
        setError(reason.message || "That meal could not be analysed.");
      } finally {
        setBusy(false);
      }
    },
    [loadDashboardData, plateDiameter, token],
  );

  const choosePhoto = useCallback(
    async (fromCamera) => {
      if (busy) return;
      if (fromCamera) {
        const permission = await Camera.requestCameraPermissionsAsync();
        if (!permission.granted) {
          Alert.alert("Camera access needed", "Enable camera access for Nutri-AI in Settings to snap a plate.");
          return;
        }
        setCameraOpen(true);
        return;
      }
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert("Photos access needed", "Enable photo access for Nutri-AI in Settings to pick a meal.");
        return;
      }
      const picked = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["images"],
        quality: 0.85,
        allowsMultipleSelection: false,
      });
      if (picked.canceled) return;
      await analyse(picked.assets?.[0]);
    },
    [analyse, busy],
  );

  const openHistoryMeal = useCallback(
    async (entry) => {
      if (!token) return;
      setBusy(true);
      setError(null);
      try {
        const result = await request(`/api/meals/${entry.meal_id}`, {}, token);
        setMeal(result);
        setScreen("results");
      } catch (reason) {
        setError(reason.message);
      } finally {
        setBusy(false);
      }
    },
    [token],
  );

  const saveCorrection = useCallback(
    async (classifiedLabel, weight) => {
      if (!correction || !meal || !token) return;
      setBusy(true);
      try {
        const body = {};
        if (classifiedLabel) body.classified_label = classifiedLabel;
        if (weight) body.estimated_weight_g = Number(weight);
        const updated = await request(
          `/api/meals/${meal.meal_id}/items/${correction.id}`,
          { method: "PATCH", body: JSON.stringify(body) },
          token,
        );
        setMeal(updated);
        setCorrection(null);
        setNotice("Correction saved. Thanks for teaching the model.");
        await loadDashboardData(token);
      } catch (reason) {
        setError(reason.message);
      } finally {
        setBusy(false);
      }
    },
    [correction, loadDashboardData, meal, token],
  );

  const updateGoal = useCallback(
    async (calorieGoal) => {
      if (!token) return;
      try {
        const fresh = await request(
          "/api/users/me/preferences",
          { method: "PATCH", body: JSON.stringify({ calorie_goal: calorieGoal }) },
          token,
        );
        setSession((current) => (current ? { ...current, user: fresh } : current));
        await persistSession(token, fresh);
        await loadDashboardData(token);
      } catch (reason) {
        setError(reason.message);
      }
    },
    [loadDashboardData, token],
  );

  if (phase === "booting") return <Splash />;
  if (phase === "onboarding") return <Onboarding onDone={finishOnboarding} />;
  if (phase === "auth")
    return (
      <AuthScreen
        onAuthenticated={enterApp}
        guestToken={upgrading ? token : null}
        onCancel={upgrading ? cancelUpgrade : null}
      />
    );

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="dark" />
      {screen === "results" && meal ? (
        <ResultsScreen
          meal={meal}
          busy={busy}
          onBack={() => setScreen("home")}
          onCorrect={setCorrection}
        />
      ) : null}
      {screen === "home" ? (
        <HomeScreen
          user={user}
          summary={summary}
          history={history}
          busy={busy}
          refreshing={refreshing}
          error={error}
          notice={notice}
          onDismissError={() => setError(null)}
          onDismissNotice={() => setNotice(null)}
          onRefresh={refresh}
          onScan={() => choosePhoto(true)}
          onPick={() => choosePhoto(false)}
          onOpenMeal={openHistoryMeal}
          onSeeAll={() => setScreen("meals")}
        />
      ) : null}
      {screen === "meals" ? (
        <MealsScreen
          history={history}
          refreshing={refreshing}
          error={error}
          onDismissError={() => setError(null)}
          onRefresh={refresh}
          onOpenMeal={openHistoryMeal}
          onScan={() => choosePhoto(true)}
        />
      ) : null}
      {screen === "stats" ? (
        <StatisticScreen
          summary={summary}
          history={history}
          refreshing={refreshing}
          error={error}
          onDismissError={() => setError(null)}
          onRefresh={refresh}
        />
      ) : null}
      {screen === "profile" ? (
        <ProfileScreen
          user={user}
          summary={summary}
          history={history}
          error={error}
          onDismissError={() => setError(null)}
          onSignOut={signOut}
          onUpgrade={beginUpgrade}
          onUpdateGoal={updateGoal}
        />
      ) : null}
      {screen === "results" ? null : (
        <BottomTabs active={screen} onNavigate={setScreen} onScan={() => choosePhoto(true)} busy={busy} />
      )}
      <CameraCapture visible={cameraOpen} onClose={() => setCameraOpen(false)} onCapture={analyse} />
      <CorrectionModal
        item={correction}
        busy={busy}
        onClose={() => setCorrection(null)}
        onSave={saveCorrection}
      />
      {busy && screen !== "results" ? (
        <View style={styles.blocker} pointerEvents="none">
          <View style={styles.blockerCard}>
            <ActivityIndicator color={C.greenDeep} />
            <Text style={styles.blockerText}>Working on it</Text>
          </View>
        </View>
      ) : null}
    </SafeAreaView>
  );
}

function Splash() {
  return (
    <View style={styles.splash}>
      <StatusBar style="dark" />
      <View style={styles.splashMark}>
        <Ionicons name="leaf" size={30} color={C.onGreen} />
      </View>
      <Text style={styles.splashTitle}>Nutri-AI</Text>
      <ActivityIndicator color={C.greenDeep} style={{ marginTop: 22 }} />
    </View>
  );
}

function BrandMark({ tone = "dark" }) {
  return (
    <View style={styles.brandRow}>
      <View style={styles.brandChip}>
        <Ionicons name="leaf" size={15} color={C.onGreen} />
      </View>
      <Text style={[styles.brandName, tone === "light" && { color: C.card }]}>Nutri-AI</Text>
    </View>
  );
}

function Segments({ total, active }) {
  return (
    <View style={styles.segments}>
      {Array.from({ length: total }, (_, index) => (
        <View key={index} style={[styles.segment, index === active && styles.segmentOn, index < active && styles.segmentDone]} />
      ))}
    </View>
  );
}

function Chip({ value, label, style }) {
  return (
    <View style={[styles.heroChip, style]}>
      <Text style={styles.heroChipValue}>{value}</Text>
      <Text style={styles.heroChipLabel}>{label}</Text>
    </View>
  );
}

function HeroCalories() {
  const radius = 76;
  const circumference = 2 * Math.PI * radius;
  const arc = (fraction) => `${circumference * fraction} ${circumference}`;
  return (
    <View style={styles.heroStage}>
      <Svg width={300} height={300}>
        <Circle cx={150} cy={150} r={128} fill={C.mintSoft} />
        <Circle cx={150} cy={150} r={112} fill={C.mint} />
        <Circle cx={150} cy={150} r={94} fill={C.card} />
        <Circle cx={150} cy={150} r={radius} stroke={C.lineSoft} strokeWidth={16} fill="none" />
        <Circle cx={150} cy={150} r={radius} stroke={C.green} strokeWidth={16} fill="none" strokeLinecap="round" strokeDasharray={arc(0.46)} rotation="-90" origin="150, 150" />
        <Circle cx={150} cy={150} r={radius} stroke={C.amber} strokeWidth={16} fill="none" strokeLinecap="round" strokeDasharray={arc(0.24)} rotation="76" origin="150, 150" />
        <Circle cx={150} cy={150} r={radius} stroke={C.blue} strokeWidth={16} fill="none" strokeLinecap="round" strokeDasharray={arc(0.13)} rotation="170" origin="150, 150" />
        <G>
          <Line x1={91} y1={91} x2={58} y2={44} stroke={C.ink} strokeWidth={1.2} opacity={0.35} />
          <Circle cx={91} cy={91} r={4.5} fill={C.ink} />
          <Line x1={226} y1={115} x2={248} y2={100} stroke={C.ink} strokeWidth={1.2} opacity={0.35} />
          <Circle cx={226} cy={115} r={4.5} fill={C.ink} />
          <Line x1={115} y1={226} x2={92} y2={260} stroke={C.ink} strokeWidth={1.2} opacity={0.35} />
          <Circle cx={115} cy={226} r={4.5} fill={C.ink} />
        </G>
      </Svg>
      <View style={styles.heroCenter} pointerEvents="none">
        <Text style={styles.heroCenterValue}>504</Text>
        <Text style={styles.heroCenterLabel}>Kcal</Text>
      </View>
      <Chip value="132 Kcal" label="Protein" style={{ left: 0, top: 4 }} />
      <Chip value="320 Kcal" label="Carbs" style={{ right: 0, top: 60 }} />
      <Chip value="52 Kcal" label="Fat" style={{ left: 12, bottom: 2 }} />
    </View>
  );
}

function HeroScan() {
  return (
    <View style={styles.heroStage}>
      <Svg width={300} height={300}>
        <Circle cx={150} cy={150} r={128} fill={C.mintSoft} />
        <Circle cx={150} cy={150} r={112} fill={C.mint} />
        <Rect x={68} y={72} width={164} height={156} rx={26} fill={C.card} />
        <Circle cx={150} cy={150} r={52} fill={C.mintSoft} />
        <Circle cx={150} cy={150} r={38} fill={C.card} />
        <Circle cx={133} cy={140} r={13} fill={C.amber} />
        <Circle cx={162} cy={137} r={10} fill={C.green} />
        <Circle cx={150} cy={165} r={11} fill={C.blue} opacity={0.8} />
        <G stroke={C.greenDeep} strokeWidth={4} strokeLinecap="round" fill="none">
          <Line x1={54} y1={92} x2={54} y2={70} />
          <Line x1={54} y1={70} x2={76} y2={70} />
          <Line x1={224} y1={70} x2={246} y2={70} />
          <Line x1={246} y1={70} x2={246} y2={92} />
          <Line x1={246} y1={208} x2={246} y2={230} />
          <Line x1={246} y1={230} x2={224} y2={230} />
          <Line x1={76} y1={230} x2={54} y2={230} />
          <Line x1={54} y1={230} x2={54} y2={208} />
        </G>
        <Line x1={72} y1={150} x2={228} y2={150} stroke={C.greenDeep} strokeWidth={2} opacity={0.45} />
      </Svg>
      <Chip value="Dal Tadka" label="94% match" style={{ left: 0, top: 8 }} />
      <Chip value="1 photo" label="12 nutrients" style={{ right: 0, bottom: 14 }} />
    </View>
  );
}

function HeroProgress() {
  const bars = [0.42, 0.58, 0.5, 0.74, 0.66, 0.9];
  return (
    <View style={styles.heroStage}>
      <Svg width={300} height={300}>
        <Circle cx={150} cy={150} r={128} fill={C.mintSoft} />
        <Circle cx={150} cy={150} r={112} fill={C.mint} />
        <Rect x={62} y={86} width={176} height={132} rx={24} fill={C.card} />
        {bars.map((value, index) => {
          const height = 78 * value;
          const x = 80 + index * 24;
          return (
            <G key={index}>
              <Rect x={x} y={116} width={13} height={78} rx={6.5} fill={C.lineSoft} />
              <Rect x={x} y={194 - height} width={13} height={height} rx={6.5} fill={index === bars.length - 1 ? C.greenDeep : C.green} />
            </G>
          );
        })}
        <Line x1={78} y1={204} x2={230} y2={204} stroke={C.line} strokeWidth={2} />
      </Svg>
      <Chip value="+18%" label="This week" style={{ right: 0, top: 22 }} />
      <Chip value="12 days" label="Streak" style={{ left: 0, bottom: 18 }} />
    </View>
  );
}

const SLIDES = [
  {
    top: "Your Daily Guide to Smarter",
    tail: "Eating.",
    icon: "leaf",
    body: "Nutri-AI reads your plate and turns it into calories, macros and micronutrients you can trust.",
    Hero: HeroCalories,
  },
  {
    top: "Snap the Plate,",
    tail: "Skip the Maths.",
    icon: "camera",
    body: "One photo is enough. We detect every item, estimate the portion and add up the nutrition for you.",
    Hero: HeroScan,
  },
  {
    top: "Progress You Can",
    tail: "Actually Feel.",
    icon: "trending-up",
    body: "Streaks, weekly trends and honest targets, so small daily choices stack into a real result.",
    Hero: HeroProgress,
  },
];

function Onboarding({ onDone }) {
  const [index, setIndex] = useState(0);
  const fade = useRef(new Animated.Value(1)).current;
  const slide = SLIDES[index];
  const Hero = slide.Hero;
  const last = index === SLIDES.length - 1;

  const go = (next) => {
    Animated.timing(fade, { toValue: 0, duration: 130, useNativeDriver: true }).start(() => {
      setIndex(next);
      Animated.timing(fade, { toValue: 1, duration: 220, useNativeDriver: true }).start();
    });
  };

  const advance = () => {
    if (last) onDone();
    else go(index + 1);
  };

  return (
    <SafeAreaView style={styles.onboard}>
      <StatusBar style="dark" />
      <View style={styles.onboardTop}>
        <BrandMark />
        <Segments total={SLIDES.length} active={index} />
      </View>
      <Animated.View style={[styles.onboardMain, { opacity: fade }]}>
        <Text style={styles.onboardTitle}>{slide.top}</Text>
        <View style={styles.onboardTitleRow}>
          <Text style={styles.onboardTitle}>{slide.tail}</Text>
          <View style={styles.onboardBadge}>
            <Ionicons name={slide.icon} size={20} color={C.onGreen} />
          </View>
        </View>
        <Text style={styles.onboardBody}>{slide.body}</Text>
        <View style={styles.onboardHero}>
          <Hero />
        </View>
      </Animated.View>
      <View style={styles.onboardFoot}>
        <Pressable style={({ pressed }) => [styles.startPill, pressed && styles.pressedSoft]} onPress={advance}>
          <View style={styles.startCircle}>
            <Ionicons name="chevron-forward" size={17} color={C.onGreen} style={{ marginRight: -9 }} />
            <Ionicons name="chevron-forward" size={17} color={C.onGreen} />
          </View>
          <Text style={styles.startText}>{last ? "Get Started" : "Continue"}</Text>
          <View style={styles.startCheck}>
            <Ionicons name="checkmark" size={16} color={C.card} />
          </View>
        </Pressable>
        {last ? (
          <Text style={styles.onboardHint}>Free to start. No card, no ads.</Text>
        ) : (
          <Pressable onPress={onDone} hitSlop={12}>
            <Text style={styles.onboardSkip}>Skip intro</Text>
          </Pressable>
        )}
      </View>
    </SafeAreaView>
  );
}

function Field({ icon, label, value, onChangeText, placeholder, secure, onToggleSecure, revealed, keyboardType, autoCapitalize, autoComplete }) {
  return (
    <View style={styles.fieldWrap}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <View style={styles.field}>
        <Ionicons name={icon} size={18} color={C.muted} />
        <TextInput
          style={styles.fieldInput}
          value={value}
          onChangeText={onChangeText}
          placeholder={placeholder}
          placeholderTextColor={C.faint}
          secureTextEntry={secure && !revealed}
          keyboardType={keyboardType || "default"}
          autoCapitalize={autoCapitalize || "none"}
          autoComplete={autoComplete}
          autoCorrect={false}
        />
        {secure ? (
          <Pressable onPress={onToggleSecure} hitSlop={10}>
            <Ionicons name={revealed ? "eye-off-outline" : "eye-outline"} size={18} color={C.muted} />
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

function AuthScreen({ onAuthenticated, guestToken, onCancel }) {
  const [mode, setMode] = useState(guestToken ? "signup" : "signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [revealed, setRevealed] = useState(false);
  const [busy, setBusy] = useState(null);
  const [failure, setFailure] = useState(null);
  const signup = mode === "signup";

  const submit = async () => {
    const trimmed = email.trim().toLowerCase();
    if (!trimmed.includes("@") || trimmed.length < 5) {
      setFailure("Enter the email address you use.");
      return;
    }
    if (password.length < 8) {
      setFailure("Passwords need at least 8 characters.");
      return;
    }
    setBusy(mode);
    setFailure(null);
    try {
      const body = signup
        ? { email: trimmed, password, name: name.trim() || null }
        : { email: trimmed, password };
      const payload = await request(
        signup ? "/api/auth/register" : "/api/auth/login",
        { method: "POST", body: JSON.stringify(body) },
        signup ? guestToken : undefined,
      );
      await onAuthenticated({ token: payload.token, user: payload.user });
    } catch (reason) {
      setFailure(reason.message);
    } finally {
      setBusy(null);
    }
  };

  const continueAsGuest = async () => {
    setBusy("guest");
    setFailure(null);
    try {
      const payload = await request("/api/auth/guest", { method: "POST" });
      await onAuthenticated({ token: payload.token, user: payload.user });
    } catch (reason) {
      setFailure(reason.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <SafeAreaView style={styles.authSafe}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <ScrollView contentContainerStyle={styles.authScroll} keyboardShouldPersistTaps="handled">
          <View style={styles.authTop}>
            <BrandMark />
            {onCancel ? (
              <Pressable style={styles.roundBtn} onPress={onCancel} hitSlop={8}>
                <Ionicons name="close" size={19} color={C.ink} />
              </Pressable>
            ) : null}
          </View>
          <Text style={styles.authTitle}>{signup ? "Create your" : "Welcome"}</Text>
          <View style={styles.onboardTitleRow}>
            <Text style={styles.authTitle}>{signup ? "account." : "back."}</Text>
            <View style={styles.onboardBadge}>
              <Ionicons name={signup ? "sparkles" : "hand-right"} size={19} color={C.onGreen} />
            </View>
          </View>
          <Text style={styles.authBody}>
            {guestToken
              ? "Add an email and password to keep the meals you already logged, on every device."
              : signup
                ? "Your plates, macros and streaks stay in sync wherever you sign in."
                : "Sign in to pick up your streak right where you left it."}
          </Text>

          <View style={styles.toggle}>
            <Pressable style={[styles.toggleBtn, !signup && styles.toggleBtnOn]} onPress={() => { setMode("signin"); setFailure(null); }}>
              <Text style={[styles.toggleText, !signup && styles.toggleTextOn]}>Sign in</Text>
            </Pressable>
            <Pressable style={[styles.toggleBtn, signup && styles.toggleBtnOn]} onPress={() => { setMode("signup"); setFailure(null); }}>
              <Text style={[styles.toggleText, signup && styles.toggleTextOn]}>Sign up</Text>
            </Pressable>
          </View>

          {signup ? (
            <Field
              icon="person-outline"
              label="Name"
              value={name}
              onChangeText={setName}
              placeholder="What should we call you?"
              autoCapitalize="words"
              autoComplete="name"
            />
          ) : null}
          <Field
            icon="mail-outline"
            label="Email"
            value={email}
            onChangeText={setEmail}
            placeholder="you@example.com"
            keyboardType="email-address"
            autoComplete="email"
          />
          <Field
            icon="lock-closed-outline"
            label="Password"
            value={password}
            onChangeText={setPassword}
            placeholder={signup ? "At least 8 characters" : "Your password"}
            secure
            revealed={revealed}
            onToggleSecure={() => setRevealed((current) => !current)}
            autoComplete={signup ? "new-password" : "current-password"}
          />

          {failure ? (
            <View style={styles.authError}>
              <Ionicons name="alert-circle" size={16} color={C.danger} />
              <Text style={styles.authErrorText}>{failure}</Text>
            </View>
          ) : null}

          <Pressable
            style={({ pressed }) => [styles.primaryBtn, pressed && styles.pressedSoft, busy && styles.btnDisabled]}
            onPress={submit}
            disabled={!!busy}
          >
            {busy === mode ? (
              <ActivityIndicator color={C.onGreen} />
            ) : (
              <>
                <Text style={styles.primaryBtnText}>{signup ? "Create account" : "Sign in"}</Text>
                <Ionicons name="arrow-forward" size={18} color={C.onGreen} />
              </>
            )}
          </Pressable>

          {guestToken ? null : (
            <>
              <View style={styles.orRow}>
                <View style={styles.orLine} />
                <Text style={styles.orText}>or</Text>
                <View style={styles.orLine} />
              </View>
              <Pressable
                style={({ pressed }) => [styles.ghostBtn, pressed && styles.pressedSoft, busy && styles.btnDisabled]}
                onPress={continueAsGuest}
                disabled={!!busy}
              >
                {busy === "guest" ? (
                  <ActivityIndicator color={C.ink} />
                ) : (
                  <>
                    <Ionicons name="flash-outline" size={17} color={C.ink} />
                    <Text style={styles.ghostBtnText}>Continue as guest</Text>
                  </>
                )}
              </Pressable>
              <Text style={styles.authFoot}>Guest meals move across when you create an account later.</Text>
            </>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function Banner({ tone, text, onDismiss }) {
  if (!text) return null;
  const danger = tone === "danger";
  return (
    <View style={[styles.banner, danger ? styles.bannerDanger : styles.bannerGood]}>
      <Ionicons name={danger ? "cloud-offline-outline" : "checkmark-circle"} size={17} color={danger ? C.danger : C.greenDeep} />
      <Text style={[styles.bannerText, danger && { color: C.danger }]} numberOfLines={3}>{text}</Text>
      <Pressable onPress={onDismiss} hitSlop={10}>
        <Ionicons name="close" size={16} color={danger ? C.danger : C.greenDeep} />
      </Pressable>
    </View>
  );
}

function ProgressRing({ progress, size, stroke, color, track, children }) {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const dash = `${Math.max(0, Math.min(1, progress)) * circumference} ${circumference}`;
  return (
    <View style={{ width: size, height: size, alignItems: "center", justifyContent: "center" }}>
      <Svg width={size} height={size} style={StyleSheet.absoluteFill}>
        <Circle cx={size / 2} cy={size / 2} r={radius} stroke={track} strokeWidth={stroke} fill="none" />
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={dash}
          rotation="-90"
          origin={`${size / 2}, ${size / 2}`}
        />
      </Svg>
      {children}
    </View>
  );
}

function TopBar({ user, onOpenCalendar, hasAlert }) {
  const hour = new Date().getHours();
  return (
    <View style={styles.topBar}>
      <View style={styles.avatar}>
        <Text style={styles.avatarText}>{initialOf(user)}</Text>
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.topGreeting}>{greetingForHour(hour)}!</Text>
        <Text style={styles.topName} numberOfLines={1}>
          {user?.name || (user?.is_guest ? "Guest" : "Friend")}
        </Text>
      </View>
      <Pressable style={({ pressed }) => [styles.roundBtn, pressed && styles.pressedSoft]} onPress={onOpenCalendar}>
        <Ionicons name="calendar-outline" size={18} color={C.ink} />
      </Pressable>
      <View style={[styles.roundBtn, { marginLeft: 10 }]}>
        <Ionicons name="notifications-outline" size={18} color={C.ink} />
        {hasAlert ? <View style={styles.dot} /> : null}
      </View>
    </View>
  );
}

function StatCard({ icon, tint, label, value, unit, sub }) {
  return (
    <View style={styles.statCard}>
      <View style={[styles.statIcon, { backgroundColor: `${tint}1f` }]}>
        <Ionicons name={icon} size={16} color={tint} />
      </View>
      <Text style={styles.statCardLabel}>{label}</Text>
      <View style={styles.statValueRow}>
        <Text style={styles.statCardValue}>{value}</Text>
        <Text style={styles.statCardUnit}>{unit}</Text>
      </View>
      <Text style={styles.statCardSub} numberOfLines={1}>{sub}</Text>
    </View>
  );
}

function CalendarStrip({ offset, selected, onSelect, onShift, marks }) {
  const days = useMemo(() => weekDays(offset), [offset]);
  const label = `${MONTHS[days[3].getMonth()]} ${days[3].getFullYear()}`;
  const todayKey = dayKey(new Date());
  return (
    <View style={styles.calCard}>
      <View style={styles.calHead}>
        <Text style={styles.calMonth}>{label}</Text>
        <View style={styles.calNav}>
          <Pressable style={({ pressed }) => [styles.calArrow, pressed && styles.pressedSoft]} onPress={() => onShift(-1)} hitSlop={6}>
            <Ionicons name="chevron-back" size={15} color={C.ink} />
          </Pressable>
          <Pressable
            style={({ pressed }) => [styles.calArrow, pressed && styles.pressedSoft, offset >= 0 && styles.calArrowOff]}
            onPress={() => onShift(1)}
            disabled={offset >= 0}
            hitSlop={6}
          >
            <Ionicons name="chevron-forward" size={15} color={offset >= 0 ? C.faint : C.ink} />
          </Pressable>
        </View>
      </View>
      <View style={styles.calRow}>
        {days.map((day, index) => {
          const key = dayKey(day);
          const active = key === selected;
          const future = day.getTime() > Date.now();
          return (
            <Pressable
              key={key}
              style={styles.calCol}
              onPress={() => onSelect(key)}
              disabled={future}
            >
              <Text style={[styles.calDow, active && styles.calDowOn, future && { color: C.faint }]}>{DOW[index]}</Text>
              <View style={[styles.calPill, active && styles.calPillOn]}>
                <Text style={[styles.calDate, active && styles.calDateOn, future && !active && { color: C.faint }]}>{day.getDate()}</Text>
              </View>
              <View style={[styles.calMark, marks.has(key) && styles.calMarkOn, key === todayKey && !marks.has(key) && styles.calMarkToday]} />
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

function ActionCard({ icon, title, sub, tint, onPress, busy }) {
  return (
    <Pressable style={({ pressed }) => [styles.actionCard, pressed && styles.pressedSoft]} onPress={onPress} disabled={busy}>
      <View style={[styles.actionIcon, { backgroundColor: tint }]}>
        <Ionicons name={icon} size={19} color={C.onGreen} />
      </View>
      <Text style={styles.actionTitle}>{busy ? "Analysing..." : title}</Text>
      <Text style={styles.actionSub} numberOfLines={1}>{sub}</Text>
    </Pressable>
  );
}

function MealRow({ entry, onPress }) {
  const thumb = imageUrl(entry.thumb_url);
  const time = parseDate(entry.captured_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const name = entry.top_items?.length ? entry.top_items.map(titleCase).join(", ") : "Logged meal";
  return (
    <Pressable style={({ pressed }) => [styles.mealRow, pressed && styles.pressedSoft]} onPress={() => onPress(entry)}>
      {thumb ? (
        <Image source={{ uri: thumb }} style={styles.mealThumb} />
      ) : (
        <View style={[styles.mealThumb, styles.mealThumbEmpty]}>
          <Ionicons name="restaurant-outline" size={19} color={C.muted} />
        </View>
      )}
      <View style={{ flex: 1 }}>
        <Text style={styles.mealName} numberOfLines={1}>{name}</Text>
        <Text style={styles.mealMeta} numberOfLines={1}>
          {`${time}  ·  ${entry.item_count} item${entry.item_count === 1 ? "" : "s"}  ·  ${formatGrams(entry.total_protein_g)} protein`}
        </Text>
      </View>
      <View style={styles.mealKcalWrap}>
        <Text style={styles.mealKcal}>{formatKcal(entry.total_calories)}</Text>
        <Text style={styles.mealKcalUnit}>Kcal</Text>
      </View>
      {entry.has_low_confidence ? <View style={styles.mealFlag} /> : null}
    </Pressable>
  );
}

function EmptyState({ icon, title, body, action, onAction }) {
  return (
    <View style={styles.empty}>
      <View style={styles.emptyIcon}>
        <Ionicons name={icon} size={24} color={C.greenDeep} />
      </View>
      <Text style={styles.emptyTitle}>{title}</Text>
      <Text style={styles.emptyBody}>{body}</Text>
      {action ? (
        <Pressable style={({ pressed }) => [styles.emptyBtn, pressed && styles.pressedSoft]} onPress={onAction}>
          <Ionicons name="camera" size={16} color={C.onGreen} />
          <Text style={styles.emptyBtnText}>{action}</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

function HomeScreen({
  user,
  summary,
  history,
  busy,
  refreshing,
  error,
  notice,
  onDismissError,
  onDismissNotice,
  onRefresh,
  onScan,
  onPick,
  onOpenMeal,
  onSeeAll,
}) {
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState(dayKey(new Date()));

  const goal = summary?.goal || {};
  const today = summary?.today || {};
  const calorieGoal = Number(goal.calories) || 2000;
  const eaten = Number(today.calories) || 0;
  const left = Math.max(0, Math.round(calorieGoal - eaten));
  const ratio = percent(eaten, calorieGoal);

  const marks = useMemo(() => new Set((history || []).map((entry) => dayKey(entry.captured_at))), [history]);
  const dayMeals = useMemo(
    () => (history || []).filter((entry) => dayKey(entry.captured_at) === selected),
    [history, selected],
  );
  const isToday = selected === dayKey(new Date());
  const dayTotal = dayMeals.reduce((sum, entry) => sum + (Number(entry.total_calories) || 0), 0);

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.screenPad}
      showsVerticalScrollIndicator={false}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.greenDeep} />}
    >
      <TopBar user={user} onOpenCalendar={onSeeAll} hasAlert={!!summary?.streak_days} />
      <Banner tone="danger" text={error} onDismiss={onDismissError} />
      <Banner tone="good" text={notice} onDismiss={onDismissNotice} />

      <View style={styles.hero}>
        <View style={styles.heroTop}>
          <View style={styles.heroPill}>
            <View style={styles.heroPillDot} />
            <Text style={styles.heroPillText}>Daily intake</Text>
          </View>
          <Text style={styles.heroDate}>{new Date().toLocaleDateString([], { day: "numeric", month: "short" })}</Text>
        </View>
        <View style={styles.heroRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.heroLabel}>Your weekly progress</Text>
            <View style={styles.heroValueRow}>
              <Text style={styles.heroValue}>{formatKcal(eaten)}</Text>
              <Text style={styles.heroGoal}>{`/ ${formatKcal(calorieGoal)} Kcal`}</Text>
            </View>
            <View style={styles.heroTrack}>
              <View style={[styles.heroFill, { width: `${Math.round(ratio * 100)}%` }]} />
            </View>
            <Text style={styles.heroFoot}>
              {`${summary?.streak_days || 0} day streak  ·  ${summary?.meals_logged || 0} meals logged`}
            </Text>
          </View>
          <ProgressRing progress={ratio} size={92} stroke={11} color={C.greenDeep} track="rgba(255,255,255,0.62)">
            <Text style={styles.ringValue}>{`${Math.round(ratio * 100)}%`}</Text>
            <Text style={styles.ringLabel}>of goal</Text>
          </ProgressRing>
        </View>
      </View>

      <View style={styles.cardRow}>
        <StatCard
          icon="flame"
          tint={C.amber}
          label="Kcal left"
          value={formatKcal(left)}
          unit="kcal"
          sub={`Target ${formatKcal(calorieGoal)}`}
        />
        <StatCard
          icon="barbell"
          tint={C.greenDeep}
          label="Protein"
          value={formatKcal(today.protein_g)}
          unit="g"
          sub={`of ${formatGrams(goal.protein_g)}`}
        />
      </View>

      <CalendarStrip
        offset={offset}
        selected={selected}
        marks={marks}
        onSelect={setSelected}
        onShift={(step) => setOffset((current) => Math.min(0, current + step))}
      />

      <View style={styles.cardRow}>
        <ActionCard icon="camera" title="Scan a plate" sub="Live camera" tint={C.mint} onPress={onScan} busy={busy} />
        <ActionCard icon="images-outline" title="From gallery" sub="Pick a photo" tint={C.mintSoft} onPress={onPick} busy={busy} />
      </View>

      <View style={styles.sectionHead}>
        <View>
          <Text style={styles.sectionTitle}>{isToday ? "Today's meals" : "Meals logged"}</Text>
          <Text style={styles.sectionSub}>
            {dayMeals.length ? `${formatKcal(dayTotal)} Kcal across ${dayMeals.length} meal${dayMeals.length === 1 ? "" : "s"}` : "Nothing logged yet"}
          </Text>
        </View>
        <Pressable onPress={onSeeAll} hitSlop={8}>
          <Text style={styles.sectionLink}>See all</Text>
        </Pressable>
      </View>

      {dayMeals.length ? (
        dayMeals.map((entry) => <MealRow key={entry.meal_id} entry={entry} onPress={onOpenMeal} />)
      ) : (
        <EmptyState
          icon="camera-outline"
          title={isToday ? "Log your first plate" : "No meals on this day"}
          body={
            isToday
              ? "Point the camera at your food. Nutri-AI handles the portion maths for you."
              : "Pick another day on the strip above, or scan something now."
          }
          action="Scan a plate"
          onAction={onScan}
        />
      )}
    </ScrollView>
  );
}

function Hatch({ radius = 11 }) {
  return (
    <View style={[styles.hatch, { borderRadius: radius }]} pointerEvents="none">
      {Array.from({ length: 12 }, (_, index) => (
        <View key={index} style={[styles.hatchLine, { top: index * 16 - 6 }]} />
      ))}
    </View>
  );
}

function BarChart({ buckets, goal, compact }) {
  const height = 148;
  const todayKey = dayKey(new Date());
  return (
    <View style={styles.chart}>
      {buckets.map((bucket) => {
        const ratio = percent(bucket.calories, goal);
        const active = bucket.date === todayKey;
        const day = new Date(`${bucket.date}T12:00:00`);
        return (
          <View key={bucket.date} style={styles.chartCol}>
            {compact ? null : (
              <Text style={[styles.chartPct, active && styles.chartPctOn]}>{`${Math.round(ratio * 100)}%`}</Text>
            )}
            <View style={[styles.chartTrack, { height, width: compact ? 12 : 22, borderRadius: compact ? 6 : 11 }]}>
              <Hatch radius={compact ? 6 : 11} />
              <View
                style={[
                  styles.chartFill,
                  {
                    height: Math.max(ratio > 0 ? 8 : 0, ratio * height),
                    borderRadius: compact ? 6 : 11,
                    backgroundColor: active ? C.greenDeep : C.green,
                  },
                ]}
              />
            </View>
            <Text style={[styles.chartDow, active && styles.chartDowOn]}>
              {compact ? day.getDate() : DOW[(day.getDay() + 6) % 7].slice(0, 1)}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

function MetricCard({ icon, tint, label, value, unit, ratio }) {
  return (
    <View style={styles.metricCard}>
      <View style={styles.metricHead}>
        <View style={[styles.metricIcon, { backgroundColor: `${tint}1f` }]}>
          <Ionicons name={icon} size={14} color={tint} />
        </View>
        <Text style={styles.metricLabel}>{label}</Text>
      </View>
      <View style={styles.statValueRow}>
        <Text style={styles.metricValue}>{value}</Text>
        <Text style={styles.metricUnit}>{unit}</Text>
      </View>
      <View style={styles.metricTrack}>
        <View style={[styles.metricFill, { width: `${Math.round(percent(ratio, 1) * 100)}%`, backgroundColor: tint }]} />
      </View>
    </View>
  );
}

function StatisticScreen({ summary, history, refreshing, error, onDismissError, onRefresh }) {
  const [range, setRange] = useState(7);
  const goal = summary?.goal || {};
  const today = summary?.today || {};
  const calorieGoal = Number(goal.calories) || 2000;
  const trend = summary?.trend || [];
  const buckets = useMemo(() => trend.slice(Math.max(0, trend.length - range)), [range, trend]);
  const logged = buckets.filter((bucket) => bucket.meals > 0);
  const average = logged.length ? logged.reduce((sum, bucket) => sum + bucket.calories, 0) / logged.length : 0;
  const delta = average > 0 ? Math.round(((today.calories || 0) - average) / Math.max(1, average) * 100) : 0;
  const best = summary?.best_day;

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.screenPad}
      showsVerticalScrollIndicator={false}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.greenDeep} />}
    >
      <View style={styles.pageHead}>
        <View>
          <Text style={styles.pageTitle}>Statistic</Text>
          <Text style={styles.pageSub}>{`${logged.length} of last ${range} days logged`}</Text>
        </View>
        <View style={styles.rangeToggle}>
          <Pressable style={[styles.rangeBtn, range === 7 && styles.rangeBtnOn]} onPress={() => setRange(7)}>
            <Text style={[styles.rangeText, range === 7 && styles.rangeTextOn]}>Week</Text>
          </Pressable>
          <Pressable style={[styles.rangeBtn, range === 14 && styles.rangeBtnOn]} onPress={() => setRange(14)}>
            <Text style={[styles.rangeText, range === 14 && styles.rangeTextOn]}>14d</Text>
          </Pressable>
        </View>
      </View>

      <Banner tone="danger" text={error} onDismiss={onDismissError} />

      <View style={styles.chartCard}>
        <View style={styles.chartHead}>
          <View>
            <Text style={styles.chartLabel}>Calories</Text>
            <View style={styles.statValueRow}>
              <Text style={styles.chartBig}>{formatKcal(today.calories)}</Text>
              <Text style={styles.chartBigUnit}>Kcal</Text>
            </View>
            <Text style={styles.chartTarget}>{`Target: ${formatKcal(calorieGoal)} Kcal`}</Text>
          </View>
          <View style={[styles.deltaChip, delta < 0 && styles.deltaChipDown]}>
            <Ionicons name={delta < 0 ? "trending-down" : "trending-up"} size={14} color={delta < 0 ? C.ink2 : C.onGreen} />
            <Text style={[styles.deltaText, delta < 0 && { color: C.ink2 }]}>{`${delta > 0 ? "+" : ""}${delta}%`}</Text>
          </View>
        </View>
        {buckets.length ? (
          <BarChart buckets={buckets} goal={calorieGoal} compact={range > 7} />
        ) : (
          <Text style={styles.chartEmpty}>Log a meal and your trend shows up here.</Text>
        )}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendSwatch, { backgroundColor: C.green }]} />
            <Text style={styles.legendText}>Eaten</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendSwatch, styles.legendGhost]}>
              <Hatch radius={4} />
            </View>
            <Text style={styles.legendText}>Target</Text>
          </View>
          <Text style={styles.legendAvg}>{`Avg ${formatKcal(average)} Kcal`}</Text>
        </View>
      </View>

      <View style={styles.grid}>
        <MetricCard
          icon="barbell"
          tint={C.greenDeep}
          label="Protein"
          value={formatKcal(today.protein_g)}
          unit={`/ ${formatKcal(goal.protein_g)} g`}
          ratio={percent(today.protein_g, goal.protein_g)}
        />
        <MetricCard
          icon="nutrition"
          tint={C.amber}
          label="Carbs"
          value={formatKcal(today.carbs_g)}
          unit={`/ ${formatKcal(goal.carbs_g)} g`}
          ratio={percent(today.carbs_g, goal.carbs_g)}
        />
        <MetricCard
          icon="water"
          tint={C.blue}
          label="Fat"
          value={formatKcal(today.fat_g)}
          unit={`/ ${formatKcal(goal.fat_g)} g`}
          ratio={percent(today.fat_g, goal.fat_g)}
        />
        <MetricCard
          icon="flame"
          tint={C.pink}
          label="Streak"
          value={`${summary?.streak_days || 0}`}
          unit="days"
          ratio={percent(summary?.streak_days || 0, 7)}
        />
      </View>

      <View style={styles.infoCard}>
        <View style={styles.infoRow}>
          <View style={[styles.metricIcon, { backgroundColor: C.mintSoft }]}>
            <Ionicons name="trophy-outline" size={14} color={C.greenDeep} />
          </View>
          <Text style={styles.infoLabel}>Best day</Text>
          <Text style={styles.infoValue}>
            {best ? `${formatKcal(best.calories)} Kcal` : "Not yet"}
          </Text>
        </View>
        <View style={styles.infoDivider} />
        <View style={styles.infoRow}>
          <View style={[styles.metricIcon, { backgroundColor: C.mintSoft }]}>
            <Ionicons name="restaurant-outline" size={14} color={C.greenDeep} />
          </View>
          <Text style={styles.infoLabel}>Meals logged</Text>
          <Text style={styles.infoValue}>{`${history?.length || 0}`}</Text>
        </View>
        <View style={styles.infoDivider} />
        <View style={styles.infoRow}>
          <View style={[styles.metricIcon, { backgroundColor: C.mintSoft }]}>
            <Ionicons name="pulse-outline" size={14} color={C.greenDeep} />
          </View>
          <Text style={styles.infoLabel}>Daily average</Text>
          <Text style={styles.infoValue}>{`${formatKcal(average)} Kcal`}</Text>
        </View>
      </View>
    </ScrollView>
  );
}

function MealsScreen({ history, refreshing, error, onDismissError, onRefresh, onOpenMeal, onScan }) {
  const groups = useMemo(() => {
    const buckets = new Map();
    (history || []).forEach((entry) => {
      const key = dayKey(entry.captured_at);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(entry);
    });
    return Array.from(buckets.entries());
  }, [history]);
  const todayKey = dayKey(new Date());

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.screenPad}
      showsVerticalScrollIndicator={false}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.greenDeep} />}
    >
      <View style={styles.pageHead}>
        <View>
          <Text style={styles.pageTitle}>Meals</Text>
          <Text style={styles.pageSub}>{`${history?.length || 0} logged so far`}</Text>
        </View>
        <Pressable style={({ pressed }) => [styles.roundBtnGreen, pressed && styles.pressedSoft]} onPress={onScan}>
          <Ionicons name="add" size={22} color={C.onGreen} />
        </Pressable>
      </View>
      <Banner tone="danger" text={error} onDismiss={onDismissError} />
      {groups.length ? (
        groups.map(([key, entries]) => {
          const date = new Date(`${key}T12:00:00`);
          const total = entries.reduce((sum, entry) => sum + (Number(entry.total_calories) || 0), 0);
          return (
            <View key={key} style={styles.group}>
              <View style={styles.groupHead}>
                <Text style={styles.groupTitle}>
                  {key === todayKey ? "Today" : date.toLocaleDateString([], { weekday: "long", day: "numeric", month: "short" })}
                </Text>
                <Text style={styles.groupTotal}>{`${formatKcal(total)} Kcal`}</Text>
              </View>
              {entries.map((entry) => (
                <MealRow key={entry.meal_id} entry={entry} onPress={onOpenMeal} />
              ))}
            </View>
          );
        })
      ) : (
        <EmptyState
          icon="restaurant-outline"
          title="No meals yet"
          body="Every plate you scan lands here with its full nutrition breakdown."
          action="Scan a plate"
          onAction={onScan}
        />
      )}
    </ScrollView>
  );
}

function DetectedPhoto({ meal, image, onSelect }) {
  const width = Number(meal.image_width) || 1000;
  const height = Number(meal.image_height) || 1000;
  return (
    <View style={[styles.photoFrame, { aspectRatio: width / height }]}>
      <Image source={{ uri: image }} style={styles.photo} resizeMode="cover" />
      <Svg style={StyleSheet.absoluteFill} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {(meal.items || []).map((item, index) => {
          if (!item.bbox) return null;
          const x = Math.max(0, Math.min(width, Number(item.bbox.x || 0) * width));
          const y = Math.max(0, Math.min(height, Number(item.bbox.y || 0) * height));
          const w = Math.max(6, Math.min(width - x, Number(item.bbox.w || 0) * width));
          const h = Math.max(6, Math.min(height - y, Number(item.bbox.h || 0) * height));
          const low = item.low_confidence && !item.user_corrected;
          const color = low ? C.amber : C.mint;
          const label = `${index + 1}. ${item.display_name}`;
          const fontSize = Math.max(14, width * 0.024);
          const labelWidth = Math.min(width - 8, Math.max(84, label.length * fontSize * 0.56 + 16));
          const labelHeight = fontSize + 10;
          const labelX = Math.max(4, Math.min(x, width - labelWidth - 4));
          const labelY = y >= labelHeight + 7 ? y - labelHeight - 3 : Math.min(height - labelHeight - 4, y + h + 4);
          return (
            <G key={item.id || index}>
              <Rect
                x={x}
                y={y}
                width={w}
                height={h}
                fill={low ? "rgba(240,163,43,0.16)" : "rgba(216,241,135,0.20)"}
                stroke={color}
                strokeWidth={Math.max(3, width * 0.006)}
                strokeDasharray={low ? "12,8" : undefined}
                onPress={() => onSelect && onSelect(item)}
              />
              <Rect
                x={labelX}
                y={labelY}
                width={labelWidth}
                height={labelHeight}
                rx={labelHeight / 2}
                fill={color}
                opacity={0.97}
                onPress={() => onSelect && onSelect(item)}
              />
              <SvgText x={labelX + 8} y={labelY + fontSize + 1} fill={C.onGreen} fontSize={fontSize} fontWeight="700">
                {label}
              </SvgText>
            </G>
          );
        })}
      </Svg>
    </View>
  );
}

function MacroPill({ label, value, tint }) {
  return (
    <View style={styles.macroPill}>
      <View style={[styles.macroPillDot, { backgroundColor: tint }]} />
      <Text style={styles.macroPillValue}>{formatGrams(value)}</Text>
      <Text style={styles.macroPillLabel}>{label}</Text>
    </View>
  );
}

function ItemCard({ item, index, onCorrect }) {
  const [open, setOpen] = useState(false);
  const low = item.low_confidence && !item.user_corrected;
  const pieces = item.piece_count && item.piece_count > 1 ? `${item.piece_count} pieces  ·  ` : "";
  return (
    <Pressable onPress={() => setOpen(!open)} style={({ pressed }) => [styles.itemCard, low && styles.itemLow, pressed && styles.pressedSoft]}>
      <View style={[styles.itemIndex, low && styles.itemIndexLow]}>
        <Text style={styles.itemIndexText}>{index + 1}</Text>
      </View>
      <View style={styles.itemMain}>
        <View style={styles.itemNameRow}>
          <Text style={styles.itemName} numberOfLines={1}>{item.display_name}</Text>
          {item.user_corrected ? (
            <View style={styles.fixedTag}>
              <Ionicons name="checkmark" size={11} color={C.greenDeep} />
              <Text style={styles.fixedTagText}>Fixed</Text>
            </View>
          ) : low ? (
            <Pressable style={styles.unsure} onPress={() => onCorrect && onCorrect(item)} hitSlop={6}>
              <Text style={styles.unsureText}>Fix</Text>
            </Pressable>
          ) : null}
        </View>
        <Text style={styles.itemMeta} numberOfLines={1}>
          {`${pieces}${formatGrams(item.estimated_weight_g)}  ·  P ${formatGrams(item.protein_g)}  C ${formatGrams(item.carbs_g)}  F ${formatGrams(item.fat_g)}`}
        </Text>
        {open ? (
          <View style={styles.itemDetail}>
            <Text style={styles.detailLabel}>CONFIDENCE</Text>
            <Text style={styles.detailValue}>{`${Math.round((Number(item.confidence) || 0) * 100)}% on this label`}</Text>
            <Text style={[styles.detailLabel, { marginTop: 10 }]}>PORTION METHOD</Text>
            <Text style={styles.detailValue}>{item.geometry?.method || item.nutrition_source || "Estimated from the photo"}</Text>
            {onCorrect ? (
              <Pressable style={styles.inlineFix} onPress={() => onCorrect(item)}>
                <Ionicons name="create-outline" size={15} color={C.greenDeep} />
                <Text style={styles.inlineFixText}>Adjust this item</Text>
              </Pressable>
            ) : null}
          </View>
        ) : null}
      </View>
      <View style={styles.itemEnergy}>
        <Text style={styles.itemKcal}>{formatKcal(item.calories)}</Text>
        <Text style={styles.itemUnit}>Kcal</Text>
      </View>
    </Pressable>
  );
}

function ResultsScreen({ meal, busy, onBack, onCorrect }) {
  const image = imageUrl(meal.image_url || meal.thumb_url);
  const totals = meal.totals || {};
  const micros = Object.entries(meal.daily_values || {}).slice(0, 6);
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.resultsPad} showsVerticalScrollIndicator={false}>
      <View style={styles.resultsTop}>
        <Pressable style={({ pressed }) => [styles.roundBtn, pressed && styles.pressedSoft]} onPress={onBack}>
          <Ionicons name="arrow-back" size={19} color={C.ink} />
        </Pressable>
        <View style={{ flex: 1, marginLeft: 12 }}>
          <Text style={styles.resultsTitle}>Meal breakdown</Text>
          <Text style={styles.resultsSub}>
            {parseDate(meal.captured_at).toLocaleString([], { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" })}
          </Text>
        </View>
        {busy ? <ActivityIndicator color={C.greenDeep} /> : null}
      </View>

      {image ? <DetectedPhoto meal={meal} image={image} onSelect={onCorrect} /> : null}

      <View style={styles.totalsCard}>
        <Text style={styles.totalsLabel}>Total energy</Text>
        <View style={styles.statValueRow}>
          <Text style={styles.totalsValue}>{formatKcal(totals.calories)}</Text>
          <Text style={styles.totalsUnit}>Kcal</Text>
        </View>
        <View style={styles.macroPillRow}>
          <MacroPill label="Protein" value={totals.protein_g} tint={C.greenDeep} />
          <MacroPill label="Carbs" value={totals.carbs_g} tint={C.amber} />
          <MacroPill label="Fat" value={totals.fat_g} tint={C.blue} />
        </View>
      </View>

      {meal.low_confidence ? (
        <View style={styles.warn}>
          <Ionicons name="alert-circle-outline" size={17} color={C.amber} />
          <Text style={styles.warnText}>
            Some items are a low-confidence guess. Tap Fix on any row to correct the label or weight.
          </Text>
        </View>
      ) : null}
      {(meal.warnings || []).map((warning, index) => (
        <View key={index} style={styles.warn}>
          <Ionicons name="information-circle-outline" size={17} color={C.ink2} />
          <Text style={styles.warnText}>{warning}</Text>
        </View>
      ))}

      <View style={styles.sectionHead}>
        <View>
          <Text style={styles.sectionTitle}>Detected items</Text>
          <Text style={styles.sectionSub}>{`${(meal.items || []).length} on this plate`}</Text>
        </View>
      </View>
      {(meal.items || []).map((item, index) => (
        <ItemCard key={item.id || index} item={item} index={index} onCorrect={onCorrect} />
      ))}

      {micros.length ? (
        <View style={styles.microCard}>
          <Text style={styles.sectionTitle}>Daily values</Text>
          {micros.map(([key, value]) => (
            <View key={key} style={styles.microRow}>
              <Text style={styles.microLabel} numberOfLines={1}>{titleCase(key)}</Text>
              <View style={styles.microTrack}>
                <View style={[styles.microFill, { width: `${Math.round(percent(value, 100) * 100)}%` }]} />
              </View>
              <Text style={styles.microValue}>{`${Math.round(Number(value) || 0)}%`}</Text>
            </View>
          ))}
        </View>
      ) : null}

      <View style={styles.engineRow}>
        <Ionicons name="hardware-chip-outline" size={14} color={C.muted} />
        <Text style={styles.engineText} numberOfLines={2}>
          {`${meal.engine || "pipeline"}  ·  plate ${meal.plate_diameter_cm || 26} cm`}
        </Text>
      </View>
    </ScrollView>
  );
}

function SettingRow({ icon, label, value, onPress, tone }) {
  const danger = tone === "danger";
  return (
    <Pressable
      style={({ pressed }) => [styles.settingRow, pressed && onPress && styles.pressedSoft]}
      onPress={onPress}
      disabled={!onPress}
    >
      <View style={[styles.settingIcon, danger && { backgroundColor: "rgba(217,83,74,0.10)" }]}>
        <Ionicons name={icon} size={16} color={danger ? C.danger : C.greenDeep} />
      </View>
      <Text style={[styles.settingLabel, danger && { color: C.danger }]}>{label}</Text>
      {value ? <Text style={styles.settingValue}>{value}</Text> : null}
      {onPress ? <Ionicons name="chevron-forward" size={16} color={C.faint} /> : null}
    </Pressable>
  );
}

const GOAL_STEPS = [1600, 1800, 2000, 2200, 2500];

function ProfileScreen({ user, summary, history, error, onDismissError, onSignOut, onUpgrade, onUpdateGoal }) {
  const goal = Math.round(Number(summary?.goal?.calories) || 2000);
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.screenPad} showsVerticalScrollIndicator={false}>
      <View style={styles.pageHead}>
        <View>
          <Text style={styles.pageTitle}>Profile</Text>
          <Text style={styles.pageSub}>{user?.is_guest ? "Guest session" : user?.email || "Signed in"}</Text>
        </View>
      </View>
      <Banner tone="danger" text={error} onDismiss={onDismissError} />

      <View style={styles.profileCard}>
        <View style={styles.profileAvatar}>
          <Text style={styles.profileAvatarText}>{initialOf(user)}</Text>
        </View>
        <Text style={styles.profileName}>{user?.name || "Guest"}</Text>
        <Text style={styles.profileMeta}>
          {`${summary?.streak_days || 0} day streak  ·  ${history?.length || 0} meals`}
        </Text>
        {user?.is_guest ? (
          <Pressable style={({ pressed }) => [styles.profileCta, pressed && styles.pressedSoft]} onPress={onUpgrade}>
            <Ionicons name="sparkles" size={16} color={C.onGreen} />
            <Text style={styles.profileCtaText}>Create an account</Text>
          </Pressable>
        ) : null}
      </View>

      <Text style={styles.groupLabel}>Daily calorie goal</Text>
      <View style={styles.goalRow}>
        {GOAL_STEPS.map((step) => (
          <Pressable
            key={step}
            style={[styles.goalChip, step === goal && styles.goalChipOn]}
            onPress={() => onUpdateGoal(step)}
          >
            <Text style={[styles.goalChipText, step === goal && styles.goalChipTextOn]}>{step}</Text>
          </Pressable>
        ))}
      </View>

      <Text style={styles.groupLabel}>Account</Text>
      <View style={styles.settingsCard}>
        <SettingRow icon="mail-outline" label="Email" value={user?.email || "Not set"} />
        <SettingRow icon="resize-outline" label="Plate size" value={`${user?.preferences?.plate_diameter_cm || 26} cm`} />
        <SettingRow icon="server-outline" label="Backend" value={API_URL.replace(/^https?:\/\//, "")} />
      </View>

      <Text style={styles.groupLabel}>Session</Text>
      <View style={styles.settingsCard}>
        <SettingRow
          icon="log-out-outline"
          label={user?.is_guest ? "End guest session" : "Sign out"}
          tone="danger"
          onPress={() =>
            Alert.alert(
              user?.is_guest ? "End guest session?" : "Sign out?",
              user?.is_guest
                ? "Guest meals stay on the server but you will need a new session to see them."
                : "You can sign back in any time.",
              [
                { text: "Cancel", style: "cancel" },
                { text: user?.is_guest ? "End session" : "Sign out", style: "destructive", onPress: onSignOut },
              ],
            )
          }
        />
      </View>
      <Text style={styles.versionText}>Nutri-AI mobile</Text>
    </ScrollView>
  );
}

function CameraCapture({ visible, onClose, onCapture }) {
  const cameraRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [shooting, setShooting] = useState(false);
  const [facing, setFacing] = useState("back");

  useEffect(() => {
    if (!visible) {
      setReady(false);
      setShooting(false);
    }
  }, [visible]);

  const shoot = async () => {
    if (!cameraRef.current || shooting) return;
    setShooting(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.6, skipProcessing: false });
      onClose();
      await onCapture(photo);
    } catch (reason) {
      console.error("[camera] capture failed:", reason);
      Alert.alert("Camera trouble", reason.message || "That shot did not come through. Try again.");
    } finally {
      setShooting(false);
    }
  };

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose} statusBarTranslucent>
      <View style={styles.camWrap}>
        <StatusBar style="light" />
        {visible ? (
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing={facing} onCameraReady={() => setReady(true)} />
        ) : null}
        <View style={styles.camTop}>
          <Pressable style={styles.camRound} onPress={onClose} hitSlop={8}>
            <Ionicons name="close" size={21} color={C.card} />
          </Pressable>
          <View style={styles.camHint}>
            <Ionicons name="scan-outline" size={14} color={C.mint} />
            <Text style={styles.camHintText}>Fill the frame with the plate</Text>
          </View>
          <Pressable style={styles.camRound} onPress={() => setFacing(facing === "back" ? "front" : "back")} hitSlop={8}>
            <Ionicons name="camera-reverse-outline" size={21} color={C.card} />
          </Pressable>
        </View>
        <View style={styles.camGuide} pointerEvents="none" />
        <View style={styles.camBottom}>
          <Text style={styles.camTip}>Shoot straight down, keep the whole plate visible.</Text>
          <Pressable style={styles.shutter} onPress={shoot} disabled={!ready || shooting}>
            <View style={[styles.shutterInner, (!ready || shooting) && { backgroundColor: C.muted }]}>
              {shooting ? <ActivityIndicator color={C.onGreen} /> : <Ionicons name="camera" size={26} color={C.onGreen} />}
            </View>
          </Pressable>
          <Text style={styles.camTip}>{ready ? "Ready" : "Warming up the camera..."}</Text>
        </View>
      </View>
    </Modal>
  );
}

function CorrectionModal({ item, busy, onClose, onSave }) {
  const [label, setLabel] = useState("");
  const [weight, setWeight] = useState("");

  useEffect(() => {
    setLabel(item ? item.classified_label || item.detected_label || "" : "");
    setWeight(item ? String(Math.round(Number(item.estimated_weight_g) || 0)) : "");
  }, [item]);

  const alternatives = (item?.alternatives || []).slice(0, 4);

  return (
    <Modal visible={!!item} animationType="slide" transparent onRequestClose={onClose}>
      <View style={styles.sheetBackdrop}>
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined}>
          <View style={styles.sheet}>
            <View style={styles.sheetGrip} />
            <Text style={styles.sheetTitle}>Fix this item</Text>
            <Text style={styles.sheetBody}>
              Corrections train the model and update this meal's totals right away.
            </Text>
            {alternatives.length ? (
              <View style={styles.altRow}>
                {alternatives.map((alt, index) => {
                  const value = alt.label || alt.name || "";
                  if (!value) return null;
                  return (
                    <Pressable key={index} style={[styles.altChip, label === value && styles.altChipOn]} onPress={() => setLabel(value)}>
                      <Text style={[styles.altChipText, label === value && styles.altChipTextOn]}>{titleCase(value)}</Text>
                    </Pressable>
                  );
                })}
              </View>
            ) : null}
            <Field icon="pricetag-outline" label="What is it really?" value={label} onChangeText={setLabel} placeholder="e.g. paneer butter masala" />
            <Field icon="scale-outline" label="Weight in grams" value={weight} onChangeText={setWeight} placeholder="e.g. 180" keyboardType="number-pad" />
            <View style={styles.sheetActions}>
              <Pressable style={({ pressed }) => [styles.ghostBtn, styles.sheetBtn, pressed && styles.pressedSoft]} onPress={onClose}>
                <Text style={styles.ghostBtnText}>Cancel</Text>
              </Pressable>
              <Pressable
                style={({ pressed }) => [styles.primaryBtn, styles.sheetBtn, pressed && styles.pressedSoft, busy && styles.btnDisabled]}
                onPress={() => onSave(label.trim(), weight.trim())}
                disabled={busy}
              >
                {busy ? <ActivityIndicator color={C.onGreen} /> : <Text style={styles.primaryBtnText}>Save fix</Text>}
              </Pressable>
            </View>
          </View>
        </KeyboardAvoidingView>
      </View>
    </Modal>
  );
}

const TABS = [
  { key: "home", icon: "home", label: "Home" },
  { key: "meals", icon: "restaurant", label: "Meals" },
  { key: "stats", icon: "stats-chart", label: "Stats" },
  { key: "profile", icon: "person", label: "You" },
];

function BottomTabs({ active, onNavigate, onScan, busy }) {
  return (
    <View style={styles.tabBar}>
      {TABS.slice(0, 2).map((tab) => (
        <TabButton key={tab.key} tab={tab} active={active === tab.key} onPress={() => onNavigate(tab.key)} />
      ))}
      <Pressable style={({ pressed }) => [styles.scanBtn, pressed && styles.pressedSoft]} onPress={onScan} disabled={busy}>
        {busy ? <ActivityIndicator color={C.onGreen} /> : <Ionicons name="scan" size={25} color={C.onGreen} />}
      </Pressable>
      {TABS.slice(2).map((tab) => (
        <TabButton key={tab.key} tab={tab} active={active === tab.key} onPress={() => onNavigate(tab.key)} />
      ))}
    </View>
  );
}

function TabButton({ tab, active, onPress }) {
  return (
    <Pressable style={styles.tabBtn} onPress={onPress} hitSlop={6}>
      <Ionicons name={active ? tab.icon : `${tab.icon}-outline`} size={21} color={active ? C.ink : C.muted} />
      <Text style={[styles.tabLabel, active && styles.tabLabelOn]}>{tab.label}</Text>
      {active ? <View style={styles.tabDot} /> : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg, paddingTop: TOP_INSET },
  screen: { flex: 1, backgroundColor: C.bg },
  screenPad: { paddingHorizontal: 18, paddingTop: 8, paddingBottom: 132 + BOTTOM_INSET },
  resultsPad: { paddingHorizontal: 18, paddingTop: 8, paddingBottom: 40 + BOTTOM_INSET },
  pressedSoft: { opacity: 0.7, transform: [{ scale: 0.985 }] },
  btnDisabled: { opacity: 0.55 },

  blocker: { ...StyleSheet.absoluteFillObject, alignItems: "center", justifyContent: "center", backgroundColor: "rgba(240,240,238,0.55)" },
  blockerCard: { backgroundColor: C.card, borderRadius: 20, paddingVertical: 20, paddingHorizontal: 26, alignItems: "center", gap: 10, ...SHADOW },
  blockerText: { color: C.ink2, fontSize: 13, fontWeight: "600" },

  splash: { flex: 1, backgroundColor: C.bg, alignItems: "center", justifyContent: "center" },
  splashMark: { width: 62, height: 62, borderRadius: 20, backgroundColor: C.mint, alignItems: "center", justifyContent: "center" },
  splashTitle: { marginTop: 16, fontSize: 22, fontWeight: "800", color: C.ink, letterSpacing: -0.4 },

  brandRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  brandChip: { width: 28, height: 28, borderRadius: 9, backgroundColor: C.mint, alignItems: "center", justifyContent: "center" },
  brandName: { fontSize: 16, fontWeight: "800", color: C.ink, letterSpacing: -0.3 },

  segments: { flexDirection: "row", alignItems: "center", gap: 5 },
  segment: { width: 8, height: 8, borderRadius: 4, backgroundColor: C.line },
  segmentOn: { width: 22, backgroundColor: C.green },
  segmentDone: { backgroundColor: C.faint },

  onboard: { flex: 1, backgroundColor: C.bg, paddingHorizontal: 22, paddingTop: TOP_INSET },
  onboardTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingTop: 12, paddingBottom: 6 },
  onboardMain: { flex: 1, paddingTop: 18 },
  onboardTitle: { fontSize: 34, lineHeight: 39, fontWeight: "800", color: C.ink, letterSpacing: -1.1 },
  onboardTitleRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  onboardBadge: { width: 38, height: 38, borderRadius: 19, backgroundColor: C.green, alignItems: "center", justifyContent: "center" },
  onboardBody: { marginTop: 12, fontSize: 14, lineHeight: 21, color: C.ink2, maxWidth: 320 },
  onboardHero: { flex: 1, alignItems: "center", justifyContent: "center", marginTop: 4 },
  onboardFoot: { paddingBottom: 16 + BOTTOM_INSET, alignItems: "center", gap: 12 },
  onboardHint: { fontSize: 12, color: C.muted },
  onboardSkip: { fontSize: 13, fontWeight: "600", color: C.ink2 },

  startPill: { alignSelf: "stretch", height: 66, borderRadius: 33, backgroundColor: C.card, flexDirection: "row", alignItems: "center", paddingLeft: 8, paddingRight: 16, ...SHADOW },
  startCircle: { width: 50, height: 50, borderRadius: 25, backgroundColor: C.green, alignItems: "center", justifyContent: "center", flexDirection: "row" },
  startText: { flex: 1, textAlign: "center", fontSize: 17, fontWeight: "800", color: C.ink, letterSpacing: -0.4 },
  startCheck: { width: 30, height: 30, borderRadius: 15, backgroundColor: C.dark, alignItems: "center", justifyContent: "center" },

  heroStage: { width: 300, height: 300, alignSelf: "center" },
  heroCenter: { position: "absolute", left: 0, right: 0, top: 120, alignItems: "center" },
  heroCenterValue: { fontSize: 34, fontWeight: "800", color: C.ink, letterSpacing: -1 },
  heroCenterLabel: { fontSize: 11, fontWeight: "700", color: C.muted, letterSpacing: 0.6 },
  heroChip: { position: "absolute", backgroundColor: C.card, borderRadius: 14, paddingVertical: 7, paddingHorizontal: 12, ...SHADOW_SOFT },
  heroChipValue: { fontSize: 13, fontWeight: "800", color: C.ink, letterSpacing: -0.3 },
  heroChipLabel: { fontSize: 10, fontWeight: "600", color: C.muted, marginTop: 1 },

  authSafe: { flex: 1, backgroundColor: C.bg, paddingTop: TOP_INSET },
  authScroll: { paddingHorizontal: 22, paddingTop: 12, paddingBottom: 40 + BOTTOM_INSET },
  authTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 26 },
  authTitle: { fontSize: 32, lineHeight: 37, fontWeight: "800", color: C.ink, letterSpacing: -1 },
  authBody: { marginTop: 10, fontSize: 13.5, lineHeight: 20, color: C.ink2, maxWidth: 330 },

  toggle: { flexDirection: "row", backgroundColor: C.lineSoft, borderRadius: 16, padding: 4, marginTop: 22, marginBottom: 6 },
  toggleBtn: { flex: 1, height: 42, borderRadius: 13, alignItems: "center", justifyContent: "center" },
  toggleBtnOn: { backgroundColor: C.card, ...SHADOW_SOFT },
  toggleText: { fontSize: 14, fontWeight: "700", color: C.muted },
  toggleTextOn: { color: C.ink },

  fieldWrap: { marginTop: 14 },
  fieldLabel: { fontSize: 11.5, fontWeight: "700", color: C.ink2, marginBottom: 7, letterSpacing: 0.2 },
  field: { flexDirection: "row", alignItems: "center", gap: 10, height: 54, borderRadius: 17, backgroundColor: C.card, borderWidth: 1, borderColor: C.line, paddingHorizontal: 15 },
  fieldInput: { flex: 1, fontSize: 15, color: C.ink, padding: 0 },

  authError: { flexDirection: "row", alignItems: "center", gap: 8, marginTop: 16, backgroundColor: "rgba(217,83,74,0.09)", borderRadius: 14, padding: 12 },
  authErrorText: { flex: 1, fontSize: 12.5, lineHeight: 18, color: C.danger, fontWeight: "600" },

  primaryBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, height: 56, borderRadius: 18, backgroundColor: C.green, marginTop: 22, ...SHADOW_SOFT },
  primaryBtnText: { fontSize: 15.5, fontWeight: "800", color: C.onGreen, letterSpacing: -0.2 },
  ghostBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, height: 56, borderRadius: 18, backgroundColor: C.card, borderWidth: 1, borderColor: C.line },
  ghostBtnText: { fontSize: 14.5, fontWeight: "700", color: C.ink },

  orRow: { flexDirection: "row", alignItems: "center", gap: 12, marginVertical: 20 },
  orLine: { flex: 1, height: 1, backgroundColor: C.line },
  orText: { fontSize: 12, fontWeight: "600", color: C.muted },
  authFoot: { marginTop: 16, fontSize: 11.5, lineHeight: 17, color: C.muted, textAlign: "center" },

  banner: { flexDirection: "row", alignItems: "center", gap: 10, borderRadius: 16, padding: 13, marginBottom: 14 },
  bannerDanger: { backgroundColor: "rgba(217,83,74,0.09)" },
  bannerGood: { backgroundColor: C.mintSoft },
  bannerText: { flex: 1, fontSize: 12.5, lineHeight: 18, color: C.greenDeep, fontWeight: "600" },

  topBar: { flexDirection: "row", alignItems: "center", gap: 12, paddingVertical: 10, marginBottom: 8 },
  avatar: { width: 44, height: 44, borderRadius: 15, backgroundColor: C.mint, alignItems: "center", justifyContent: "center" },
  avatarText: { fontSize: 17, fontWeight: "800", color: C.onGreen },
  topGreeting: { fontSize: 12, color: C.muted, fontWeight: "600" },
  topName: { fontSize: 17, fontWeight: "800", color: C.ink, letterSpacing: -0.4, marginTop: 1 },
  roundBtn: { width: 40, height: 40, borderRadius: 14, backgroundColor: C.card, alignItems: "center", justifyContent: "center", ...SHADOW_SOFT },
  roundBtnGreen: { width: 44, height: 44, borderRadius: 15, backgroundColor: C.green, alignItems: "center", justifyContent: "center", ...SHADOW_SOFT },
  dot: { position: "absolute", top: 9, right: 10, width: 7, height: 7, borderRadius: 4, backgroundColor: C.green, borderWidth: 1.5, borderColor: C.card },

  hero: { backgroundColor: C.mint, borderRadius: 26, padding: 18, ...SHADOW },
  heroTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 14 },
  heroPill: { flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: "rgba(255,255,255,0.75)", borderRadius: 12, paddingVertical: 5, paddingHorizontal: 10 },
  heroPillDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: C.greenDeep },
  heroPillText: { fontSize: 11.5, fontWeight: "700", color: C.onGreen },
  heroDate: { fontSize: 12, fontWeight: "700", color: C.greenDeep },
  heroRow: { flexDirection: "row", alignItems: "center", gap: 14 },
  heroLabel: { fontSize: 13, fontWeight: "700", color: C.greenDeep },
  heroValueRow: { flexDirection: "row", alignItems: "flex-end", gap: 6, marginTop: 4 },
  heroValue: { fontSize: 36, fontWeight: "800", color: C.onGreen, letterSpacing: -1.4, lineHeight: 40 },
  heroGoal: { fontSize: 12.5, fontWeight: "700", color: C.greenDeep, paddingBottom: 5 },
  heroTrack: { height: 7, borderRadius: 4, backgroundColor: "rgba(255,255,255,0.66)", marginTop: 10, overflow: "hidden" },
  heroFill: { height: 7, borderRadius: 4, backgroundColor: C.greenDeep },
  heroFoot: { fontSize: 11.5, fontWeight: "600", color: C.greenDeep, marginTop: 9 },
  ringValue: { fontSize: 17, fontWeight: "800", color: C.onGreen, letterSpacing: -0.5 },
  ringLabel: { fontSize: 9.5, fontWeight: "700", color: C.greenDeep, marginTop: 1 },

  cardRow: { flexDirection: "row", gap: 12, marginTop: 14 },
  statCard: { flex: 1, backgroundColor: C.card, borderRadius: 20, padding: 15, ...SHADOW_SOFT },
  statIcon: { width: 30, height: 30, borderRadius: 11, alignItems: "center", justifyContent: "center", marginBottom: 11 },
  statCardLabel: { fontSize: 11.5, fontWeight: "700", color: C.muted },
  statValueRow: { flexDirection: "row", alignItems: "flex-end", gap: 4 },
  statCardValue: { fontSize: 25, fontWeight: "800", color: C.ink, letterSpacing: -0.9, lineHeight: 29 },
  statCardUnit: { fontSize: 11.5, fontWeight: "700", color: C.muted, paddingBottom: 4 },
  statCardSub: { fontSize: 11, color: C.faint, fontWeight: "600", marginTop: 3 },

  calCard: { backgroundColor: C.card, borderRadius: 22, padding: 15, marginTop: 14, ...SHADOW_SOFT },
  calHead: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 12 },
  calMonth: { fontSize: 14.5, fontWeight: "800", color: C.ink, letterSpacing: -0.3 },
  calNav: { flexDirection: "row", gap: 8 },
  calArrow: { width: 28, height: 28, borderRadius: 10, backgroundColor: C.lineSoft, alignItems: "center", justifyContent: "center" },
  calArrowOff: { opacity: 0.5 },
  calRow: { flexDirection: "row", justifyContent: "space-between" },
  calCol: { alignItems: "center", flex: 1 },
  calDow: { fontSize: 10.5, fontWeight: "700", color: C.muted, marginBottom: 7 },
  calDowOn: { color: C.ink },
  calPill: { width: 34, height: 40, borderRadius: 14, alignItems: "center", justifyContent: "center", backgroundColor: C.lineSoft },
  calPillOn: { backgroundColor: C.dark },
  calDate: { fontSize: 14, fontWeight: "700", color: C.ink },
  calDateOn: { color: C.card },
  calMark: { width: 5, height: 5, borderRadius: 3, marginTop: 7, backgroundColor: "transparent" },
  calMarkOn: { backgroundColor: C.green },
  calMarkToday: { backgroundColor: C.line },

  actionCard: { flex: 1, backgroundColor: C.card, borderRadius: 20, padding: 15, ...SHADOW_SOFT },
  actionIcon: { width: 36, height: 36, borderRadius: 13, alignItems: "center", justifyContent: "center", marginBottom: 11 },
  actionTitle: { fontSize: 13.5, fontWeight: "800", color: C.ink, letterSpacing: -0.2 },
  actionSub: { fontSize: 11, color: C.muted, fontWeight: "600", marginTop: 2 },

  sectionHead: { flexDirection: "row", alignItems: "flex-end", justifyContent: "space-between", marginTop: 24, marginBottom: 12 },
  sectionTitle: { fontSize: 16.5, fontWeight: "800", color: C.ink, letterSpacing: -0.4 },
  sectionSub: { fontSize: 11.5, color: C.muted, fontWeight: "600", marginTop: 3 },
  sectionLink: { fontSize: 12.5, fontWeight: "700", color: C.greenDeep },

  mealRow: { flexDirection: "row", alignItems: "center", gap: 12, backgroundColor: C.card, borderRadius: 20, padding: 12, marginBottom: 10, ...SHADOW_SOFT },
  mealThumb: { width: 52, height: 52, borderRadius: 16, backgroundColor: C.lineSoft },
  mealThumbEmpty: { alignItems: "center", justifyContent: "center" },
  mealName: { fontSize: 14, fontWeight: "800", color: C.ink, letterSpacing: -0.2 },
  mealMeta: { fontSize: 11, color: C.muted, fontWeight: "600", marginTop: 3 },
  mealKcalWrap: { alignItems: "flex-end" },
  mealKcal: { fontSize: 16, fontWeight: "800", color: C.ink, letterSpacing: -0.4 },
  mealKcalUnit: { fontSize: 9.5, fontWeight: "700", color: C.muted },
  mealFlag: { position: "absolute", top: 12, left: 12, width: 9, height: 9, borderRadius: 5, backgroundColor: C.amber, borderWidth: 2, borderColor: C.card },

  empty: { backgroundColor: C.card, borderRadius: 22, padding: 24, alignItems: "center", ...SHADOW_SOFT },
  emptyIcon: { width: 52, height: 52, borderRadius: 18, backgroundColor: C.mintSoft, alignItems: "center", justifyContent: "center" },
  emptyTitle: { fontSize: 15.5, fontWeight: "800", color: C.ink, marginTop: 14, letterSpacing: -0.3 },
  emptyBody: { fontSize: 12.5, lineHeight: 19, color: C.ink2, textAlign: "center", marginTop: 6, maxWidth: 260 },
  emptyBtn: { flexDirection: "row", alignItems: "center", gap: 7, backgroundColor: C.green, borderRadius: 15, paddingVertical: 11, paddingHorizontal: 18, marginTop: 16 },
  emptyBtnText: { fontSize: 13, fontWeight: "800", color: C.onGreen },

  pageHead: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingVertical: 12, marginBottom: 10 },
  pageTitle: { fontSize: 27, fontWeight: "800", color: C.ink, letterSpacing: -0.9 },
  pageSub: { fontSize: 12, color: C.muted, fontWeight: "600", marginTop: 3 },
  rangeToggle: { flexDirection: "row", backgroundColor: C.lineSoft, borderRadius: 13, padding: 3 },
  rangeBtn: { paddingVertical: 7, paddingHorizontal: 13, borderRadius: 11 },
  rangeBtnOn: { backgroundColor: C.card, ...SHADOW_SOFT },
  rangeText: { fontSize: 12, fontWeight: "700", color: C.muted },
  rangeTextOn: { color: C.ink },

  chartCard: { backgroundColor: C.card, borderRadius: 24, padding: 18, ...SHADOW },
  chartHead: { flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between" },
  chartLabel: { fontSize: 12.5, fontWeight: "700", color: C.muted },
  chartBig: { fontSize: 34, fontWeight: "800", color: C.ink, letterSpacing: -1.3, lineHeight: 38 },
  chartBigUnit: { fontSize: 13, fontWeight: "700", color: C.muted, paddingBottom: 5 },
  chartTarget: { fontSize: 11.5, fontWeight: "600", color: C.faint, marginTop: 3 },
  deltaChip: { flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: C.mint, borderRadius: 12, paddingVertical: 6, paddingHorizontal: 10 },
  deltaChipDown: { backgroundColor: C.lineSoft },
  deltaText: { fontSize: 12, fontWeight: "800", color: C.onGreen },
  chartEmpty: { fontSize: 12.5, color: C.muted, textAlign: "center", paddingVertical: 40 },

  chart: { flexDirection: "row", alignItems: "flex-end", justifyContent: "space-between", marginTop: 22 },
  chartCol: { alignItems: "center", flex: 1 },
  chartPct: { fontSize: 10, fontWeight: "800", color: C.muted, marginBottom: 7 },
  chartPctOn: { color: C.ink },
  chartTrack: { backgroundColor: C.lineSoft, overflow: "hidden", justifyContent: "flex-end" },
  chartFill: { width: "100%" },
  chartDow: { fontSize: 10.5, fontWeight: "700", color: C.muted, marginTop: 9 },
  chartDowOn: { color: C.ink },
  hatch: { ...StyleSheet.absoluteFillObject, overflow: "hidden" },
  hatchLine: { position: "absolute", left: -18, width: 60, height: 1.5, backgroundColor: C.line, transform: [{ rotate: "-45deg" }] },

  legend: { flexDirection: "row", alignItems: "center", gap: 16, marginTop: 18, paddingTop: 14, borderTopWidth: 1, borderTopColor: C.lineSoft },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 6 },
  legendSwatch: { width: 12, height: 12, borderRadius: 4 },
  legendGhost: { backgroundColor: C.lineSoft, overflow: "hidden" },
  legendText: { fontSize: 11, fontWeight: "700", color: C.muted },
  legendAvg: { flex: 1, textAlign: "right", fontSize: 11, fontWeight: "700", color: C.ink2 },

  grid: { flexDirection: "row", flexWrap: "wrap", gap: 12, marginTop: 14 },
  metricCard: { width: "47.6%", flexGrow: 1, backgroundColor: C.card, borderRadius: 20, padding: 15, ...SHADOW_SOFT },
  metricHead: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 11 },
  metricIcon: { width: 26, height: 26, borderRadius: 9, alignItems: "center", justifyContent: "center" },
  metricLabel: { fontSize: 12, fontWeight: "700", color: C.ink2 },
  metricValue: { fontSize: 22, fontWeight: "800", color: C.ink, letterSpacing: -0.8, lineHeight: 26 },
  metricUnit: { fontSize: 10.5, fontWeight: "700", color: C.muted, paddingBottom: 4 },
  metricTrack: { height: 6, borderRadius: 3, backgroundColor: C.lineSoft, marginTop: 10, overflow: "hidden" },
  metricFill: { height: 6, borderRadius: 3 },

  infoCard: { backgroundColor: C.card, borderRadius: 22, paddingHorizontal: 16, marginTop: 14, ...SHADOW_SOFT },
  infoRow: { flexDirection: "row", alignItems: "center", gap: 11, paddingVertical: 15 },
  infoLabel: { flex: 1, fontSize: 13.5, fontWeight: "700", color: C.ink },
  infoValue: { fontSize: 13, fontWeight: "800", color: C.ink2 },
  infoDivider: { height: 1, backgroundColor: C.lineSoft },

  group: { marginBottom: 16 },
  groupHead: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 10 },
  groupTitle: { fontSize: 14, fontWeight: "800", color: C.ink, letterSpacing: -0.3 },
  groupTotal: { fontSize: 12, fontWeight: "700", color: C.greenDeep },
  groupLabel: { fontSize: 11.5, fontWeight: "800", color: C.muted, letterSpacing: 0.5, marginTop: 24, marginBottom: 10, textTransform: "uppercase" },

  resultsTop: { flexDirection: "row", alignItems: "center", paddingVertical: 10, marginBottom: 12 },
  resultsTitle: { fontSize: 18, fontWeight: "800", color: C.ink, letterSpacing: -0.5 },
  resultsSub: { fontSize: 11.5, color: C.muted, fontWeight: "600", marginTop: 2 },
  photoFrame: { width: "100%", borderRadius: 24, overflow: "hidden", backgroundColor: C.lineSoft, ...SHADOW },
  photo: { width: "100%", height: "100%" },

  totalsCard: { backgroundColor: C.mint, borderRadius: 24, padding: 18, marginTop: 14, ...SHADOW },
  totalsLabel: { fontSize: 12.5, fontWeight: "700", color: C.greenDeep },
  totalsValue: { fontSize: 38, fontWeight: "800", color: C.onGreen, letterSpacing: -1.5, lineHeight: 42 },
  totalsUnit: { fontSize: 13, fontWeight: "700", color: C.greenDeep, paddingBottom: 6 },
  macroPillRow: { flexDirection: "row", gap: 8, marginTop: 16 },
  macroPill: { flex: 1, backgroundColor: "rgba(255,255,255,0.78)", borderRadius: 15, paddingVertical: 11, alignItems: "center" },
  macroPillDot: { width: 7, height: 7, borderRadius: 4, marginBottom: 6 },
  macroPillValue: { fontSize: 14, fontWeight: "800", color: C.ink, letterSpacing: -0.3 },
  macroPillLabel: { fontSize: 10, fontWeight: "700", color: C.ink2, marginTop: 2 },

  warn: { flexDirection: "row", alignItems: "flex-start", gap: 9, backgroundColor: C.card, borderRadius: 16, padding: 13, marginTop: 12, ...SHADOW_SOFT },
  warnText: { flex: 1, fontSize: 12, lineHeight: 18, color: C.ink2, fontWeight: "600" },

  itemCard: { flexDirection: "row", alignItems: "flex-start", gap: 11, backgroundColor: C.card, borderRadius: 20, padding: 13, marginBottom: 10, ...SHADOW_SOFT },
  itemLow: { borderWidth: 1, borderColor: "rgba(240,163,43,0.45)" },
  itemIndex: { width: 26, height: 26, borderRadius: 9, backgroundColor: C.mint, alignItems: "center", justifyContent: "center", marginTop: 1 },
  itemIndexLow: { backgroundColor: "rgba(240,163,43,0.22)" },
  itemIndexText: { fontSize: 11.5, fontWeight: "800", color: C.onGreen },
  itemMain: { flex: 1 },
  itemNameRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  itemName: { flex: 1, fontSize: 14, fontWeight: "800", color: C.ink, letterSpacing: -0.2 },
  unsure: { backgroundColor: C.amber, borderRadius: 9, paddingVertical: 3, paddingHorizontal: 9 },
  unsureText: { fontSize: 10.5, fontWeight: "800", color: C.onGreen },
  fixedTag: { flexDirection: "row", alignItems: "center", gap: 3, backgroundColor: C.mintSoft, borderRadius: 9, paddingVertical: 3, paddingHorizontal: 8 },
  fixedTagText: { fontSize: 10, fontWeight: "800", color: C.greenDeep },
  itemMeta: { fontSize: 11, color: C.muted, fontWeight: "600", marginTop: 4 },
  itemDetail: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: C.lineSoft },
  detailLabel: { fontSize: 9.5, fontWeight: "800", color: C.faint, letterSpacing: 0.7 },
  detailValue: { fontSize: 12.5, color: C.ink2, fontWeight: "600", marginTop: 3 },
  inlineFix: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 12 },
  inlineFixText: { fontSize: 12.5, fontWeight: "800", color: C.greenDeep },
  itemEnergy: { alignItems: "flex-end" },
  itemKcal: { fontSize: 15.5, fontWeight: "800", color: C.ink, letterSpacing: -0.4 },
  itemUnit: { fontSize: 9.5, fontWeight: "700", color: C.muted },

  microCard: { backgroundColor: C.card, borderRadius: 22, padding: 17, marginTop: 14, ...SHADOW_SOFT },
  microRow: { flexDirection: "row", alignItems: "center", gap: 10, marginTop: 13 },
  microLabel: { width: 92, fontSize: 11.5, fontWeight: "700", color: C.ink2 },
  microTrack: { flex: 1, height: 7, borderRadius: 4, backgroundColor: C.lineSoft, overflow: "hidden" },
  microFill: { height: 7, borderRadius: 4, backgroundColor: C.green },
  microValue: { width: 40, textAlign: "right", fontSize: 11.5, fontWeight: "800", color: C.ink },

  engineRow: { flexDirection: "row", alignItems: "center", gap: 7, marginTop: 18, paddingHorizontal: 4 },
  engineText: { flex: 1, fontSize: 10.5, color: C.muted, fontWeight: "600" },

  profileCard: { backgroundColor: C.card, borderRadius: 24, padding: 22, alignItems: "center", ...SHADOW },
  profileAvatar: { width: 66, height: 66, borderRadius: 23, backgroundColor: C.mint, alignItems: "center", justifyContent: "center" },
  profileAvatarText: { fontSize: 26, fontWeight: "800", color: C.onGreen },
  profileName: { fontSize: 19, fontWeight: "800", color: C.ink, marginTop: 13, letterSpacing: -0.5 },
  profileMeta: { fontSize: 12, fontWeight: "600", color: C.muted, marginTop: 4 },
  profileCta: { flexDirection: "row", alignItems: "center", gap: 7, backgroundColor: C.green, borderRadius: 15, paddingVertical: 11, paddingHorizontal: 18, marginTop: 16 },
  profileCtaText: { fontSize: 13, fontWeight: "800", color: C.onGreen },

  goalRow: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  goalChip: { backgroundColor: C.card, borderRadius: 14, paddingVertical: 10, paddingHorizontal: 14, borderWidth: 1, borderColor: C.line },
  goalChipOn: { backgroundColor: C.green, borderColor: C.green },
  goalChipText: { fontSize: 13, fontWeight: "700", color: C.ink2 },
  goalChipTextOn: { color: C.onGreen, fontWeight: "800" },

  settingsCard: { backgroundColor: C.card, borderRadius: 22, paddingHorizontal: 16, ...SHADOW_SOFT },
  settingRow: { flexDirection: "row", alignItems: "center", gap: 11, paddingVertical: 15 },
  settingIcon: { width: 30, height: 30, borderRadius: 11, backgroundColor: C.mintSoft, alignItems: "center", justifyContent: "center" },
  settingLabel: { flex: 1, fontSize: 13.5, fontWeight: "700", color: C.ink },
  settingValue: { fontSize: 12, fontWeight: "600", color: C.muted, maxWidth: 150 },
  versionText: { fontSize: 11, color: C.faint, textAlign: "center", marginTop: 26, fontWeight: "600" },

  camWrap: { flex: 1, backgroundColor: "#000000" },
  camTop: { position: "absolute", top: 30 + (TOP_INSET || 24), left: 18, right: 18, flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 10 },
  camRound: { width: 42, height: 42, borderRadius: 15, backgroundColor: "rgba(255,255,255,0.18)", alignItems: "center", justifyContent: "center" },
  camHint: { flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: "rgba(0,0,0,0.45)", borderRadius: 14, paddingVertical: 8, paddingHorizontal: 12 },
  camHintText: { fontSize: 11.5, fontWeight: "700", color: C.card },
  camGuide: { position: "absolute", top: "24%", left: "12%", right: "12%", aspectRatio: 1, borderRadius: 28, borderWidth: 2, borderColor: "rgba(216,241,135,0.7)" },
  camBottom: { position: "absolute", bottom: 44 + BOTTOM_INSET, left: 18, right: 18, alignItems: "center", gap: 14 },
  camTip: { fontSize: 11.5, color: "rgba(255,255,255,0.78)", fontWeight: "600", textAlign: "center" },
  shutter: { width: 82, height: 82, borderRadius: 41, borderWidth: 3, borderColor: "rgba(255,255,255,0.5)", alignItems: "center", justifyContent: "center" },
  shutterInner: { width: 66, height: 66, borderRadius: 33, backgroundColor: C.mint, alignItems: "center", justifyContent: "center" },

  sheetBackdrop: { flex: 1, backgroundColor: "rgba(20,22,18,0.42)", justifyContent: "flex-end" },
  sheet: { backgroundColor: C.bg, borderTopLeftRadius: 28, borderTopRightRadius: 28, padding: 22, paddingBottom: 34 + BOTTOM_INSET },
  sheetGrip: { alignSelf: "center", width: 42, height: 4, borderRadius: 2, backgroundColor: C.line, marginBottom: 18 },
  sheetTitle: { fontSize: 20, fontWeight: "800", color: C.ink, letterSpacing: -0.5 },
  sheetBody: { fontSize: 12.5, lineHeight: 19, color: C.ink2, marginTop: 6 },
  altRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 14 },
  altChip: { backgroundColor: C.card, borderRadius: 13, paddingVertical: 9, paddingHorizontal: 13, borderWidth: 1, borderColor: C.line },
  altChipOn: { backgroundColor: C.green, borderColor: C.green },
  altChipText: { fontSize: 12.5, fontWeight: "700", color: C.ink2 },
  altChipTextOn: { color: C.onGreen, fontWeight: "800" },
  sheetActions: { flexDirection: "row", gap: 12, marginTop: 6 },
  sheetBtn: { flex: 1, marginTop: 22 },

  tabBar: { position: "absolute", bottom: 18 + BOTTOM_INSET, left: 18, right: 18, height: 72, borderRadius: 26, backgroundColor: C.card, flexDirection: "row", alignItems: "center", paddingHorizontal: 8, ...SHADOW },
  tabBtn: { flex: 1, alignItems: "center", justifyContent: "center", gap: 3 },
  tabLabel: { fontSize: 9.5, fontWeight: "700", color: C.muted },
  tabLabelOn: { color: C.ink },
  tabDot: { position: "absolute", top: 5, width: 16, height: 3, borderRadius: 2, backgroundColor: C.green },
  scanBtn: { width: 62, height: 62, borderRadius: 22, backgroundColor: C.green, alignItems: "center", justifyContent: "center", marginHorizontal: 6, ...SHADOW },
});
