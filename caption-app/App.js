import { useState } from 'react';
import {
  StyleSheet, Text, View, TouchableOpacity,
  Image, ActivityIndicator, ScrollView, Alert
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';

// ← Your computer's local IP
const API_URL = 'http://192.168.29.11:5000';

export default function App() {
  const [media, setMedia]       = useState(null);    // { uri, type: 'image'|'video' }
  const [caption, setCaption]   = useState('');
  const [frames, setFrames]     = useState(null);    // frames analyzed for video
  const [loading, setLoading]   = useState(false);
  const [history, setHistory]   = useState([]);

  // Pick image from gallery
  async function pickImage() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Allow access to your photos.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8,
    });
    if (!result.canceled) {
      setMedia({ uri: result.assets[0].uri, type: 'image' });
      setCaption('');
      setFrames(null);
    }
  }

  // Pick video from gallery
  async function pickVideo() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Allow access to your gallery.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 0.8,
      videoMaxDuration: 60,   // max 60 seconds
    });
    if (!result.canceled) {
      setMedia({ uri: result.assets[0].uri, type: 'video' });
      setCaption('');
      setFrames(null);
    }
  }

  // Take photo with camera
  async function takePhoto() {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Allow camera access.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) {
      setMedia({ uri: result.assets[0].uri, type: 'image' });
      setCaption('');
      setFrames(null);
    }
  }

  // Send to API
  async function generate() {
    if (!media) return;
    setLoading(true);
    setCaption('');
    setFrames(null);

    try {
      const formData = new FormData();
      const isVideo  = media.type === 'video';

      formData.append(isVideo ? 'video' : 'image', {
        uri:  media.uri,
        type: isVideo ? 'video/mp4' : 'image/jpeg',
        name: isVideo ? 'video.mp4' : 'photo.jpg',
      });

      const endpoint = isVideo ? '/caption_video' : '/caption';
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = await response.json();

      if (data.error) {
        Alert.alert('Error', data.error);
      } else {
        setCaption(data.caption);
        if (data.frames_analyzed) setFrames(data.frames_analyzed);
        setHistory(prev =>
          [{ uri: media.uri, type: media.type, caption: data.caption }, ...prev].slice(0, 10)
        );
      }
    } catch (err) {
      Alert.alert('Connection Error',
        `Could not reach API at ${API_URL}\nMake sure the Flask server is running.`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <StatusBar style="light" />

      {/* Header */}
      <Text style={styles.title}>Caption Generator</Text>
      <Text style={styles.subtitle}>AI-powered image & video captioning</Text>

      {/* Preview box */}
      <View style={styles.previewBox}>
        {media ? (
          <>
            <Image
              source={{ uri: media.uri }}
              style={styles.preview}
            />
            {media.type === 'video' && (
              <View style={styles.videoBadge}>
                <Text style={styles.videoBadgeText}>🎬 VIDEO</Text>
              </View>
            )}
          </>
        ) : (
          <Text style={styles.placeholder}>No media selected</Text>
        )}
      </View>

      {/* Buttons */}
      <View style={styles.row}>
        <TouchableOpacity style={styles.btn} onPress={pickImage}>
          <Text style={styles.btnText}>� Image</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.btn} onPress={pickVideo}>
          <Text style={styles.btnText}>🎬 Video</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.btn} onPress={takePhoto}>
          <Text style={styles.btnText}>📸 Camera</Text>
        </TouchableOpacity>
      </View>

      {/* Generate button */}
      <TouchableOpacity
        style={[styles.generateBtn, !media && styles.disabled]}
        onPress={generate}
        disabled={!media || loading}
      >
        {loading
          ? <ActivityIndicator color="#fff" />
          : <Text style={styles.generateText}>
              {media?.type === 'video' ? 'Analyse Video' : 'Generate Caption'}
            </Text>
        }
      </TouchableOpacity>

      {/* Loading hint for video */}
      {loading && media?.type === 'video' && (
        <Text style={styles.hint}>Analysing video frames, this may take a moment...</Text>
      )}

      {/* Caption result */}
      {caption !== '' && (
        <View style={styles.captionBox}>
          <View style={styles.captionHeader}>
            <Text style={styles.captionLabel}>Caption</Text>
            {frames && (
              <Text style={styles.framesText}>{frames} frames analysed</Text>
            )}
          </View>
          <Text style={styles.captionText}>{caption}</Text>
        </View>
      )}

      {/* History */}
      {history.length > 0 && (
        <View style={styles.historySection}>
          <Text style={styles.historyTitle}>Recent</Text>
          {history.map((item, idx) => (
            <View key={idx} style={styles.historyItem}>
              <View style={styles.historyThumb}>
                <Image source={{ uri: item.uri }} style={styles.historyImage} />
                {item.type === 'video' && (
                  <Text style={styles.historyVideoIcon}>🎬</Text>
                )}
              </View>
              <Text style={styles.historyCaption}>{item.caption}</Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f0f',
  },
  content: {
    padding: 24,
    paddingTop: 60,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 28,
  },
  previewBox: {
    width: '100%',
    height: 260,
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  preview: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  videoBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: '#00000099',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  videoBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  placeholder: {
    color: '#444',
    fontSize: 15,
  },
  row: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 16,
    width: '100%',
  },
  btn: {
    flex: 1,
    backgroundColor: '#1e1e1e',
    padding: 13,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2e2e2e',
  },
  btnText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '500',
  },
  generateBtn: {
    width: '100%',
    backgroundColor: '#6c63ff',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 10,
  },
  disabled: {
    backgroundColor: '#333',
  },
  generateText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  hint: {
    color: '#666',
    fontSize: 12,
    marginBottom: 16,
    textAlign: 'center',
  },
  captionBox: {
    width: '100%',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 28,
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#6c63ff44',
  },
  captionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  captionLabel: {
    color: '#6c63ff',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  framesText: {
    color: '#555',
    fontSize: 11,
  },
  captionText: {
    color: '#fff',
    fontSize: 17,
    lineHeight: 26,
  },
  historySection: {
    width: '100%',
  },
  historyTitle: {
    color: '#888',
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 12,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  historyItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 10,
    marginBottom: 10,
    gap: 12,
  },
  historyThumb: {
    position: 'relative',
  },
  historyImage: {
    width: 56,
    height: 56,
    borderRadius: 8,
  },
  historyVideoIcon: {
    position: 'absolute',
    bottom: 2,
    right: 2,
    fontSize: 12,
  },
  historyCaption: {
    flex: 1,
    color: '#ccc',
    fontSize: 13,
    lineHeight: 20,
  },
});
