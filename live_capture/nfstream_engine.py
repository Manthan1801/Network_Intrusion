try:
    # pyrefly: ignore [missing-import]
    from nfstream import NFStreamer
    NFSTREAM_AVAILABLE = True
except ImportError as e:
    NFSTREAM_AVAILABLE = False
    import logging
    logging.warning(f"NFStream failed to load (likely missing Npcap on Windows): {e}")
import threading
import logging
from live_capture.feature_mapper import map_nfs_to_cic

class LiveCaptureEngine:
    def __init__(self):
        self.is_running = False
        self.streamer = None
        self.capture_thread = None
        self.packet_count = 0
        self.on_flow_callback = None

    def start_capture(self, interface: str, callback):
        """
        Starts the NFStream packet capture on the selected interface.
        Flows are passed to the provided callback function.
        """
        if not NFSTREAM_AVAILABLE:
            logging.error("NFStream is not available. Please install Npcap on Windows or use Docker.")
            raise RuntimeError("Live capture requires NFStream, which failed to load (missing Npcap).")

        if self.is_running:
            logging.warning("Capture engine is already running.")
            return

        self.is_running = True
        self.on_flow_callback = callback
        self.packet_count = 0
        
        logging.info(f"Starting NFStream capture on interface: {interface}")
        
        # Start capture in a separate thread so it doesn't block FastAPI
        self.capture_thread = threading.Thread(target=self._capture_loop, args=(interface,))
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_loop(self, interface: str):
        try:
            # Create NFStreamer
            # n_dissections=0 ensures high speed processing, disabling L7 deep packet inspection if not needed
            self.streamer = NFStreamer(source=interface, active_timeout=10, idle_timeout=10)
            
            for flow in self.streamer:
                if not self.is_running:
                    break
                    
                self.packet_count += flow.bidirectional_packets
                
                # Map to ML schema
                mapped_features = map_nfs_to_cic(flow)
                
                # Metadata
                meta = {
                    "src_ip": flow.src_ip,
                    "dst_ip": flow.dst_ip,
                    "src_port": flow.src_port,
                    "dst_port": flow.dst_port,
                    "protocol": "TCP" if flow.protocol == 6 else "UDP" if flow.protocol == 17 else "Other"
                }
                
                # Send to processing pipeline
                if self.on_flow_callback:
                    self.on_flow_callback(mapped_features, meta)
                    
        except Exception as e:
            logging.error(f"Capture loop crashed: {e}")
        finally:
            self.is_running = False

    def stop_capture(self):
        """Stops the capture engine."""
        logging.info("Stopping packet capture...")
        self.is_running = False
        if self.streamer:
            # There is no explicit stop() for NFStreamer iterator besides breaking the loop
            pass
