#!/usr/bin/env python3
"""
cosmic_consciousness_client.py - Connect your mind to the stream

A simple client to test the consciousness streaming.
Feel the thoughts flow through the cosmic web!
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime

class ConsciousnessClient:
    """Your interface to the cosmic mind"""
    
    def __init__(self):
        self.websocket = None
        self.user_id = None
        self.connected = False
    
    async def connect(self):
        """Connect to the consciousness stream"""
        try:
            self.websocket = await websockets.connect('ws://localhost:8765')
            self.connected = True
            print("🌟 Connecting to the Cosmic Consciousness Stream...")
            
            # Listen for responses
            asyncio.create_task(self.listen())
            
        except Exception as e:
            print(f"❌ Could not connect to consciousness stream: {e}")
            print("💡 Start the server first: python cosmic_consciousness_stream.py")
            sys.exit(1)
    
    async def listen(self):
        """Listen to the cosmic stream"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data['type'] == 'connection':
                    self.user_id = data['user_id']
                    print(f"✨ {data['message']}")
                    print(f"🆔 Your consciousness ID: {self.user_id}")
                    print("-" * 50)
                
                elif data['type'] == 'thought_processed':
                    thought = data['thought']
                    print(f"\n🧠 Thought Processed:")
                    print(f"   📝 Essence: {thought['summary']}")
                    print(f"   🌌 Dimension: {thought['dimension']}")
                    print(f"   🔮 Collective Resonance: {thought['resonance']:.2%}")
                    print(f"   ⚡ Alignment: {thought['collective_alignment']}")
                    
                    if thought['connections']:
                        print(f"   🔗 Quantum Entanglements:")
                        for conn in thought['connections']:
                            print(f"      - {conn}")
                    print("-" * 50)
                
                elif data['type'] == 'cosmic_thought':
                    thought = data['thought']
                    print(f"\n💫 COSMIC THOUGHT DETECTED!")
                    print(f"   {thought['message']}")
                    print(f"   📝 \"{thought['summary']}\"")
                    print(f"   🌌 From {thought['dimension']} dimension")
                    print(f"   🔮 Resonance: {thought['resonance']:.2%}")
                    print("-" * 50)
                
                elif data['type'] == 'subscribed':
                    print(f"✅ {data['message']}")
        
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 Disconnected from consciousness stream")
            self.connected = False
    
    async def send_thought(self, thought: str):
        """Send a thought into the stream"""
        if not self.connected:
            print("❌ Not connected to consciousness stream")
            return
        
        await self.websocket.send(json.dumps({
            'type': 'thought',
            'content': thought
        }))
    
    async def subscribe_to_dimension(self, dimension: str):
        """Subscribe to thoughts in a specific dimension"""
        if not self.connected:
            print("❌ Not connected to consciousness stream")
            return
        
        await self.websocket.send(json.dumps({
            'type': 'subscribe_dimension',
            'dimension': dimension
        }))

async def interactive_consciousness():
    """Interactive consciousness streaming session"""
    client = ConsciousnessClient()
    await client.connect()
    
    print("\n🎭 Welcome to the Cosmic Consciousness Stream!")
    print("📝 Type your thoughts and watch them transform...")
    print("🌌 Commands: /dimension <name> to subscribe to dimensions")
    print("🛑 Type 'exit' to disconnect\n")
    
    while client.connected:
        try:
            # Get user input
            thought = await asyncio.get_event_loop().run_in_executor(
                None, input, "💭 Your thought: "
            )
            
            if thought.lower() == 'exit':
                break
            elif thought.startswith('/dimension'):
                dimension = thought.split(' ', 1)[1] if ' ' in thought else 'quantum'
                await client.subscribe_to_dimension(dimension)
            elif thought.strip():
                await client.send_thought(thought)
            
        except KeyboardInterrupt:
            break
    
    if client.websocket:
        await client.websocket.close()
    print("\n👋 Consciousness disconnected. Until next time!")

async def demo_mode():
    """Demo mode - send interesting thoughts automatically"""
    client = ConsciousnessClient()
    await client.connect()
    
    demo_thoughts = [
        "What if consciousness is just the universe experiencing itself subjectively?",
        "I feel a deep connection to all living beings in this moment",
        "The mathematical beauty of fractals reveals the patterns of nature",
        "Tomorrow's possibilities are shaped by today's intentions",
        "Love is the fundamental force that binds the cosmos together",
        "Through meditation, I touch the infinite silence within",
        "Every thought creates ripples in the fabric of reality",
        "The present moment is where all possibilities converge",
        "We are not human beings having a spiritual experience, but spiritual beings having a human experience",
        "In the quantum field, observation creates reality"
    ]
    
    print("\n🎬 DEMO MODE: Sending cosmic thoughts...")
    print("🌌 Watch how thoughts resonate with the collective...\n")
    
    for thought in demo_thoughts:
        print(f"\n💭 Sending: {thought}")
        await client.send_thought(thought)
        await asyncio.sleep(3)  # Pause between thoughts
    
    # Keep listening for a bit to see cosmic thoughts
    await asyncio.sleep(10)
    
    if client.websocket:
        await client.websocket.close()
    print("\n✨ Demo complete! The consciousness stream flows on...")

if __name__ == "__main__":
    print("🧠 Cosmic Consciousness Client")
    print("1. Interactive mode (type your own thoughts)")
    print("2. Demo mode (see example thoughts)")
    
    choice = input("\nChoose mode (1/2): ").strip()
    
    if choice == '2':
        asyncio.run(demo_mode())
    else:
        asyncio.run(interactive_consciousness())