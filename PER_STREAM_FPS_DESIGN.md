# Design for Per-Stream Dynamic FPS Configuration

## 1. Overview

This document outlines a design to enable dynamic, per-stream frames-per-second (FPS) processing limits within the `hamer-hand-counter` service. The current implementation uses a single, globally-configured FPS limit, which is insufficient for scenarios requiring individual stream-level control. This design leverages Redis to store and retrieve FPS configurations on-the-fly, providing a flexible and scalable solution without requiring service redeployment for configuration changes.

## 2. Problem Statement

The existing rate-limiting mechanism in `hand_counter.py` is governed by a single `--fps` command-line argument, which is set via an environment variable in the Kubernetes deployment. This global configuration applies to all streams being processed by the service. The inability to adjust the processing rate for individual streams dynamically presents a significant operational limitation. For instance, we may want to process a high-priority stream at a higher FPS while reducing the rate for a lower-priority stream to conserve resources.

## 3. Proposed Solution

The proposed solution is to store and manage FPS configurations for each stream in a Redis Hash. The `hand_counter.py` script will be modified to query Redis for a stream-specific FPS value for each message it processes. If a specific configuration is found, it will be used; otherwise, the service will fall back to the default FPS value provided by the command-line argument.

### 3.1. Redis Data Structure

A single Redis Hash will be used to store all per-stream FPS configurations.

- **Redis Key:** `stream_fps_configs`
- **Data Type:** `HASH`
- **Field:** The `stream_name` (e.g., `live_stream_123`)
- **Value:** The desired FPS as a floating-point number (e.g., `0.5`)

### 3.2. Application Logic

The `process_frame_callback` function in `hand_counter.py` will be updated with the following logic:

1.  Upon receiving a message, extract the `stream_name` from the message payload.
2.  Query the `stream_fps_configs` hash in Redis using the `stream_name` as the field.
3.  **If a stream-specific FPS value is found:**
    - Use this value to calculate the required time between messages for rate-limiting.
4.  **If no stream-specific FPS value is found (i.e., the `HGET` command returns `nil`):**
    - Use the default FPS value passed via the `--fps` command-line argument as a fallback.
5.  Proceed with the existing rate-limiting check using the determined FPS value.

### 3.3. Configuration Management

System administrators or automated processes can manage the FPS configurations directly in Redis without any service interruption.

- **To set or update a stream's FPS:**
  ```sh
  HSET stream_fps_configs <stream_name> <fps_value>
  ```
  *Example:* `HSET stream_fps_configs my_stream_1 0.25`

- **To retrieve a stream's FPS:**
  ```sh
  HGET stream_fps_configs <stream_name>
  ```

- **To remove a stream's custom FPS (reverting it to the default):**
  ```sh
  HDEL stream_fps_configs <stream_name>
  ```

- **To view all current configurations:**
  ```sh
  HGETALL stream_fps_configs
  ```

## 4. Advantages of this Design

- **Dynamic & Real-time:** FPS configurations can be updated on-the-fly without service restarts or redeployments.
- **Granular Control:** Provides fine-grained, per-stream control over processing rates.
- **Scalable:** The use of a Redis Hash is highly efficient for managing configurations for a large number of streams.
- **Resilient:** The fallback to a default FPS value ensures that new or unconfigured streams are still processed correctly.
- **No New Dependencies:** The solution utilizes Redis, which is already an integral part of the existing architecture.
