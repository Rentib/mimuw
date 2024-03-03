/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.solution;

import java.util.Map;

import cp2023.base.ComponentId;
import cp2023.base.DeviceId;
import cp2023.base.StorageSystem;

public final class StorageSystemFactory {
  public static StorageSystem newSystem(
      Map<DeviceId, Integer> deviceTotalSlots,
      Map<ComponentId, DeviceId> componentPlacement
  ) {
    if (deviceTotalSlots == null) {
      throw new IllegalArgumentException("deviceTotalSlots is null");
    }
    if (deviceTotalSlots.isEmpty()) {
      throw new IllegalArgumentException("deviceTotalSlots is empty");
    }
    for (var entry : deviceTotalSlots.entrySet()) {
      var device = entry.getKey();
      var capacity = entry.getValue();

      if (device == null || capacity == null) {
        throw new IllegalArgumentException("Device or capacity is null");
      }

      if (capacity <= 0) {
        throw new IllegalArgumentException("Device capacity must be positive");
      }

      int assignedComponents = (int) componentPlacement.values().stream()
        .filter(id -> id.equals(device))
        .count();
      if (assignedComponents > capacity) {
        throw new IllegalArgumentException("Too many components assigned to a device");
      }
    }

    if (componentPlacement == null) {
      throw new IllegalArgumentException("componentPlacement is null");
    }
    for (var entry : componentPlacement.entrySet()) {
      var component = entry.getKey();
      var device = entry.getValue();

      if (component == null || device == null) {
        throw new IllegalArgumentException("Component or device is null");
      }

      if (!deviceTotalSlots.containsKey(device)) {
        throw new IllegalArgumentException("Device not present in deviceTotalSlots");
      }
    }

    return new StorageSystemImpl(deviceTotalSlots, componentPlacement);
  }
}
