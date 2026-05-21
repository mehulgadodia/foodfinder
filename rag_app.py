import streamlit as st
import requests
import time
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CACHED_CITIES = {
    "mumbai": (18.9388, 72.8354),
    "delhi": (28.6139, 77.2090),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "surat": (21.1702, 72.8311),
    "kolkata": (22.5726, 88.3639),
    "indore": (22.7196, 75.8577),
    "coimbatore": (11.0168, 76.9558),
    "kochi": (9.9312, 76.2673),
    "nagpur": (21.1458, 79.0882),
    "kanpur": (26.4499, 80.3319),
    "patna": (25.5941, 85.1376),
    "bhopal": (23.2599, 77.4130),
    "visakhapatnam": (17.6868, 83.2185),
    "vadodara": (22.3072, 73.1812),
    "chandigarh": (30.7333, 76.7794),
    "mangalore": (12.9141, 74.8560),
    "mysore": (12.2958, 76.6414),
    "trivandrum": (8.5241, 76.9366),
    "goa": (15.2993, 74.1240),
    "nashik": (19.9975, 73.7898),
    "aurangabad": (19.8762, 75.3433),
    "rajkot": (22.2731, 70.7517),
    "vijayawada": (16.5062, 80.6417),
    "guwahati": (26.1445, 91.7892),
    "dehradun": (30.3165, 78.0322),
    "varanasi": (25.3176, 82.9739),
    "agra": (27.1767, 78.0081),
    "amritsar": (31.6340, 74.8723),
    " ludhiana": (30.9010, 75.8573),
    "jodhpur": (26.2389, 73.0243),
    "ranchi": (23.3441, 85.3095),
    "jamshedpur": (22.8003, 86.2029),
}

def log_step(message):
    if 'log' not in st.session_state:
        st.session_state.log = []
    st.session_state.log.append(message)

def geocode_city(city, state):
    log_step(f"🔍 Geocoding: {city}, {state}")
    
    if 'geocode_cache' not in st.session_state:
        st.session_state.geocode_cache = {}
    
    city_lower = city.lower().strip()
    cache_key = f"{city_lower}_{state}"
    
    if cache_key in st.session_state.geocode_cache:
        lat, lon = st.session_state.geocode_cache[cache_key]
        log_step(f"✅ Using cached coordinates: ({lat}, {lon})")
        return lat, lon
    
    if city_lower in CACHED_CITIES:
        lat, lon = CACHED_CITIES[city_lower]
        log_step(f"✅ Using cached coordinates: ({lat}, {lon})")
        return lat, lon
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': f"{city}, {state}, India",
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'FoodFinderApp/1.0',
        'Accept': 'application/json',
        'Referer': 'http://localhost'
    }
    
    for attempt in range(3):
        try:
            log_step(f"📡 Request attempt {attempt + 1}/3...")
            response = requests.get(url, params=params, headers=headers, timeout=30)
            log_step(f"📥 Nominatim API response: {response.status_code}")
            
            if response.status_code == 403 or response.status_code == 429:
                log_step("⚠️ Rate limited - waiting before retry...")
                time.sleep(min(30, 2 ** attempt * 3))
                continue
                
            if response.status_code == 406:
                log_step("❌ Not acceptable - check User-Agent")
                st.error("Geocoding service error. Please try a different city.")
                return None, None
                
            if response.status_code != 200:
                log_step(f"❌ API error: {response.status_code} - {response.text[:100]}")
                st.error(f"Geocoding failed with status {response.status_code}")
                return None, None
            
            data = response.json()
            
            if not data:
                log_step("❌ No results for this city")
                st.warning(f"City '{city}' not found. Try a different spelling or major city.")
                return None, None
                
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            log_step(f"✅ Coordinates: ({lat}, {lon})")
            st.session_state.geocode_cache[cache_key] = (lat, lon)
            return lat, lon
            
        except requests.exceptions.ConnectionError:
            log_step("❌ Network error - check internet connection")
            st.error("Cannot connect to geocoding service. Check your internet connection.")
            break
        except requests.exceptions.Timeout:
            log_step(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt < 2:
                time.sleep(1)
                continue
            st.error("Geocoding timed out. Please try again.")
        except ValueError as e:
            log_step(f"❌ Data parsing error: {e}")
            st.error("Invalid response from geocoding service.")
            break
        except Exception as e:
            log_step(f"❌ Unexpected error: {type(e).__name__}: {e}")
            st.error(f"Geocoding error: {e}")
            break
    
    return None, None

class RestaurantRAG:
    def __init__(self):
        log_step("📦 Initializing RAG engine...")
        log_step("   Loading embedding model: all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        log_step("   ✅ Model loaded")
        
        log_step("💾 Connecting to ChromaDB...")
        self.client = chromadb.Client(Settings(
            persist_directory="./foodfinder_db",
            anonymized_telemetry=False
        ))
        log_step("   ✅ ChromaDB connected")
        
        self.restaurants = []
        
    def fetch_restaurants_from_osm(self, lat, lon, radius=2000, max_results=500):
        log_step(f"🌐 Fetching restaurants from OpenStreetMap...")
        log_step(f"   📍 Location: ({lat}, {lon})")
        log_step(f"   📏 Radius: {radius}m")
        
        if radius > 3000:
            log_step(f"⚠️ Large radius detected. Capping results to {max_results} restaurants.")
        
        overpass_urls = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
        
        query = f"""
        [out:json][timeout:120][maxsize:1073741824];
        (
          node["amenity"="restaurant"](around:{radius},{lat},{lon});
          way["amenity"="restaurant"](around:{radius},{lat},{lon});
        );
        out body {max_results};
        """
        
        last_error = None
        for url_idx, overpass_url in enumerate(overpass_urls):
            for attempt in range(3):
                try:
                    log_step(f"📤 Trying endpoint {url_idx + 1}/3 (attempt {attempt + 1}/3)...")
                    log_step(f"   ⏳ This may take 15-45 seconds...")
                    
                    response = requests.post(
                        overpass_url, 
                        data={'data': query}, 
                        timeout=120,
                        headers={'User-Agent': 'FoodFinderApp/1.0 (educational)'}
                    )
                    
                    log_step(f"📥 Response status: {response.status_code}")
                    
                    if response.status_code == 429:
                        log_step("⚠️ Rate limited - switching to next endpoint...")
                        wait_time = min(60, 2 ** attempt * 5)
                        log_step(f"   ⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        last_error = "Rate limited"
                        continue
                        
                    if response.status_code == 504:
                        log_step("⚠️ Gateway timeout - trying next endpoint...")
                        last_error = "Timeout 504"
                        continue
                        
                    if response.status_code == 403:
                        log_step("⚠️ Forbidden - trying next endpoint...")
                        last_error = "Forbidden"
                        continue
                        
                    if response.status_code != 200:
                        log_step(f"❌ HTTP error: {response.status_code}")
                        last_error = f"HTTP {response.status_code}"
                        break
                    
                    content = response.text.strip()
                    
                    if not content:
                        log_step("❌ Empty response from server")
                        last_error = "Empty response"
                        return []
                    
                    if len(content) < 10:
                        log_step(f"❌ Response too short: {content}")
                        last_error = "Invalid response"
                        return []
                    
                    if content.startswith('<'):
                        log_step("❌ Received HTML error page")
                        last_error = "HTML response"
                        return []
                    
                    log_step("📝 Parsing JSON response...")
                    data = response.json()
                    
                    if 'elements' not in data:
                        log_step(f"❌ Invalid response structure: {list(data.keys())}")
                        last_error = "Missing elements"
                        return []
                    
                    elements = data.get('elements', [])
                    log_step(f"📊 Raw elements received: {len(elements)}")
                    
                    if len(elements) == 0:
                        log_step("⚠️ No restaurants found in this area")
                        st.info("No restaurants found in this area. Try increasing the radius or a different location.")
                        self.restaurants = []
                        return []
                    
                    restaurants = []
                    for i, element in enumerate(elements):
                        try:
                            if element.get('type') == 'node':
                                lat_val = element.get('lat')
                                lon_val = element.get('lon')
                            elif element.get('type') == 'way' and 'center' in element:
                                lat_val = element['center'].get('lat')
                                lon_val = element['center'].get('lon')
                            else:
                                continue
                            
                            if lat_val is None or lon_val is None:
                                continue
                            
                            tags = element.get('tags', {})
                            name = tags.get('name', 'Unnamed Restaurant')
                            
                            rating = tags.get('rating:median') or tags.get('rating') or tags.get('score') or 'Unknown'
                            
                            restaurants.append({
                                'id': str(element.get('id', f'rest_{i}')),
                                'name': name,
                                'cuisine': tags.get('cuisine', 'Unknown'),
                                'lat': lat_val,
                                'lon': lon_val,
                                'address': f"{tags.get('addr:street', '')} {tags.get('addr:housenumber', '')}".strip(),
                                'opening_hours': tags.get('opening_hours', 'Unknown'),
                                'website': tags.get('website', 'Unknown'),
                                'phone': tags.get('phone', 'Unknown'),
                                'price_level': tags.get('price_level', 'Unknown'),
                                'rating': rating
                            })
                            
                            if i < 5:
                                log_step(f"   🏪 {name}")
                        except Exception as e:
                            log_step(f"   ⚠️ Skipping element {i}: {e}")
                            continue
                    
                    self.restaurants = restaurants
                    log_step(f"✅ Successfully extracted {len(restaurants)} restaurants")
                    return restaurants
                    
                except requests.exceptions.Timeout:
                    log_step(f"⏰ Request timed out on attempt {attempt + 1}")
                    if attempt < 2:
                        log_step("   Retrying in 3 seconds...")
                        time.sleep(3)
                        continue
                    last_error = "Timeout"
                except requests.exceptions.ConnectionError:
                    log_step("❌ Network error - cannot reach Overpass API")
                    last_error = "Connection error"
                    break
                except ValueError as e:
                    log_step(f"❌ JSON parsing error: {e}")
                    last_error = f"JSON error: {e}"
                    break
                except Exception as e:
                    log_step(f"❌ Unexpected error: {type(e).__name__}: {e}")
                    last_error = f"{type(e).__name__}: {e}"
                    break
        
        log_step("❌ All Overpass endpoints failed")
        if last_error:
            st.error(f"Overpass API unavailable ({last_error}). Try again in a few minutes or use a different location.")
        
        return []
    
    def index_restaurants(self, batch_size=100):
        if not self.restaurants:
            log_step("⚠️ No restaurants to index")
            return
            
        log_step(f"🔢 Indexing {len(self.restaurants)} restaurants...")
        
        try:
            collection = self.client.get_or_create_collection("restaurants")
            
            log_step("📄 Creating text documents...")
            texts = [self._create_doc(r) for r in self.restaurants]
            
            log_step("⏳ Generating embeddings (this may take a moment)...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
            log_step(f"   ✅ Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")
            
            ids = [r['id'] for r in self.restaurants]
            metadatas = [{k: str(v) for k, v in r.items()} for r in self.restaurants]
            
            log_step("💾 Upserting to ChromaDB in batches...")
            
            # Process in batches to handle large datasets
            for i in range(0, len(texts), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_texts = texts[i:i+batch_size]
                
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
                log_step(f"   📦 Uploaded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            log_step("✅ Indexing complete!")
            
        except Exception as e:
            log_step(f"❌ Indexing error: {type(e).__name__}: {e}")
            st.error(f"Failed to index restaurants: {e}")
    
    def _create_doc(self, r):
        return f"""
        Restaurant: {r['name']}
        Cuisine: {r['cuisine']}
        Rating: {r.get('rating', 'Unknown')}
        Address: {r['address']}
        Hours: {r['opening_hours']}
        Phone: {r['phone']}
        Website: {r['website']}
        Price Level: {r['price_level']}
        """
    
    def retrieve(self, query, top_k=5, min_rating=None):
        log_step(f"🔎 Searching for: '{query}'")
        
        try:
            collection = self.client.get_or_create_collection("restaurants")
            
            log_step("⏳ Encoding query...")
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            log_step(f"📊 Querying vector DB for top {top_k} results...")
            
            # Build filter for minimum rating
            where_clause = None
            if min_rating:
                try:
                    rating_val = float(min_rating)
                    where_clause = {"rating": {"$gte": str(rating_val)}}
                except:
                    pass
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k * 2,  # Get more to filter
                where=where_clause
            )
            
            # Sort by rating if we have ratings
            if results and results.get('metadatas') and results['metadatas'][0]:
                rated_results = []
                for i, meta in enumerate(results['metadatas'][0]):
                    try:
                        rating = float(meta.get('rating', 0))
                        if rating > 0:
                            rated_results.append({
                                'doc': results['documents'][0][i],
                                'meta': meta,
                                'dist': results['distances'][0][i],
                                'rating': rating
                            })
                    except:
                        continue
                
                if rated_results:
                    rated_results.sort(key=lambda x: x['rating'], reverse=True)
                    results['documents'] = [[r['doc'] for r in rated_results[:top_k]]]
                    results['metadatas'] = [[r['meta'] for r in rated_results[:top_k]]]
                    results['distances'] = [[r['dist'] for r in rated_results[:top_k]]]
            
            if results and results.get('documents') and len(results['documents'][0]) > 0:
                log_step(f"✅ Found {len(results['documents'][0])} relevant restaurants")
            else:
                log_step("⚠️ No matching restaurants found")
            
            return results
            
        except Exception as e:
            log_step(f"❌ Search error: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def generate_response(self, query, use_ollama=True, min_rating=None):
        results = self.retrieve(query, min_rating=min_rating)
        
        context = ""
        if results and results.get('documents') and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                context += f"\n\nOption {i+1}:\n{doc}"
        
        if use_ollama:
            try:
                import ollama
                log_step("🤖 Generating AI response with Ollama (llama3.2)...")
                
                prompt = f"""Based on the following restaurant information, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

                response = ollama.chat(
                    model='llama3.2',
                    messages=[{'role': 'user', 'content': prompt}]
                )
                log_step("✅ AI response generated")
                return response['message']['content'], results
                
            except ImportError:
                log_step("⚠️ Ollama not installed - showing raw results")
                st.info("Ollama not found. Install it to get AI-powered responses.")
            except Exception as e:
                log_step(f"⚠️ Ollama error: {e}")
                st.warning("Ollama not available. Showing raw search results.")
        
        log_step("📋 Formatting results...")
        return self._format_results(results), results
    
    def _format_results(self, results):
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No restaurants found matching your criteria."
        
        response = "Here are the best matching restaurants:\n\n"
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            response += f"**{i+1}. {metadata.get('name', 'Unknown')}**\n"
            response += f"   Cuisine: {metadata.get('cuisine', 'Unknown')}\n"
            response += f"   Address: {metadata.get('address', 'Unknown')}\n"
            response += f"   Hours: {metadata.get('opening_hours', 'Unknown')}\n\n"
        
        return response

st.set_page_config(page_title="FoodFinder RAG", page_icon="🍔", layout="wide")

st.title("🍔 FoodFinder - AI-Powered Restaurant Search")
st.markdown("*Powered by open-source models - No API keys needed!*")

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RestaurantRAG()
    st.session_state.log = []

col1, col2 = st.columns([1, 2])

with st.sidebar:
    st.header("📋 Workflow Monitor")
    st.markdown("### How it works:")
    st.markdown("""
    **1. Geocoding**  
    Convert city name to lat/lon using Nominatim API
    
    **2. Data Fetch**  
    Query Overpass API for restaurants in radius
    
    **3. Embedding**  
    Convert text to vectors using sentence-transformers
    
    **4. Indexing**  
    Store in ChromaDB for fast similarity search
    
    **5. Retrieval**  
    Find similar restaurants by query
    
    **6. Response**  
    Display results (or enhance with Ollama LLM)
    """)
    
    st.markdown("---")
    st.subheader("🔄 Process Log")
    if st.button("🗑️ Clear Log"):
        st.session_state.log = []
    
    log_container = st.container()
    with log_container:
        if st.session_state.get('log'):
            for entry in st.session_state.log[-25:]:
                st.caption(entry)
        else:
            st.caption("⬆️ Click 'Find Restaurants' to start...")

with col1:
    st.header("📍 Location")
    city_input = st.text_input("City", value=st.session_state.get('city_value', "Mumbai"), key="city_input")
    state = st.selectbox("State", [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi"
    ], index=st.session_state.get('state_idx', 14), key="state_select")
    radius = st.slider("Radius (m)", 500, 5000, 2000)
    
    if radius > 3000:
        st.warning("Large radius may take longer. Results capped at 500 restaurants.")
    
    if st.button("🔍 Find Restaurants", type="primary"):
        with st.spinner("Searching..."):
            lat, lon = geocode_city(city_input, state)
            if lat and lon:
                max_results = 500 if radius > 3000 else 200
                restaurants = st.session_state.rag_engine.fetch_restaurants_from_osm(lat, lon, radius, max_results)
                if restaurants:
                    st.session_state.rag_engine.index_restaurants()
                    st.success(f"Found {len(restaurants)} restaurants in {city_input}, {state}!")
            else:
                st.error("Could not find that city. Please try another.")
    
    st.markdown("### Popular Cities")
    cities = [
        ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bangalore", "Karnataka"),
        ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"), ("Kolkata", "West Bengal"),
        ("Pune", "Maharashtra"), ("Ahmedabad", "Gujarat"), ("Jaipur", "Rajasthan"), ("Lucknow", "Uttar Pradesh")
    ]
    
    for c, s in cities:
        if st.button(f"📍 {c}"):
            st.session_state.city_value = c
            state_list = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi"
    ]
            st.session_state.state_idx = state_list.index(s)
            st.rerun()

with col2:
    st.header("🤖 Ask about restaurants")
    
    if not st.session_state.rag_engine.restaurants:
        st.info("👈 Search for restaurants first to enable AI queries!")
    else:
        user_query = st.text_area(
            "What would you like to know?",
            placeholder="e.g., Find Italian restaurants that are open now",
            height=100
        )
        
        col_rating, col_ollama = st.columns([1, 1])
        with col_rating:
            min_rating = st.selectbox("Minimum Rating", ["Any", "3+ stars", "4+ stars", "4.5+ stars"])
        with col_ollama:
            use_ollama = st.checkbox("Use AI (requires Ollama)", value=True)
        
        if st.button("Ask AI", type="primary") and user_query:
            rating_filter = None
            if min_rating == "3+ stars":
                rating_filter = 3
            elif min_rating == "4+ stars":
                rating_filter = 4
            elif min_rating == "4.5+ stars":
                rating_filter = 4.5
            
            with st.spinner("Thinking..."):
                response, results = st.session_state.rag_engine.generate_response(
                    user_query, 
                    use_ollama=use_ollama,
                    min_rating=rating_filter
                )
                
                st.markdown("### 💡 Answer")
                st.write(response)
                
                with st.expander("See retrieved restaurants"):
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i]
                        rating = metadata.get('rating', 'Unknown')
                        rating_display = f"⭐ {rating}" if rating != 'Unknown' else "No rating"
                        st.write(f"**{metadata.get('name', 'Unknown')}** {rating_display}")
                        st.caption(f"{metadata.get('cuisine', 'Unknown')} | {metadata.get('address', 'Unknown')}")

st.markdown("---")
st.markdown("### 🛠️ Tech Stack (All Free)")
cols = st.columns(4)
with cols[0]:
    st.info("**Data**\nOpenStreetMap\nOverpass API")
with cols[1]:
    st.info("**Embeddings**\nSentence-Transformers\nall-MiniLM-L6-v2")
with cols[2]:
    st.info("**Vector DB**\nChromaDB")
with cols[3]:
    st.info("**LLM**\nOllama (optional)\nLocal models")